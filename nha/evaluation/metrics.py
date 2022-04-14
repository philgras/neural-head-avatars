from pytorch_msssim import ms_ssim
import numpy as np
import torch
import face_alignment
import lpips
import cpbd
import sys

sys.path.append("./deps/")

# storing information if optimum of metric is high or low
metrics_opt_dict = dict(L1="-", L2="-", PSNR="+", MS_SSIM="+", LMK="-", LPIPS="-", CPBD="+")


class Metric:
    OPTIMUM_MAX = '+'
    OPTIMUM_MIN = '-'

    def __init__(self, optimum, device='cuda'):
        """
        :param optimum: either Metric.OPTIMUM_MAX or Metric.OPTIMUM_MIN
        :param device:
        """
        self._device = device
        self._optimum = optimum

    def get_optimum(self):
        return self._optimum


class CPBD(Metric):
    """
    Evaluates the overall sharpness of an image
    """

    def __init__(self, device='cuda'):
        super().__init__(Metric.OPTIMUM_MAX, device)

    def __call__(self, x, scale='symmetric'):
        """
        assumes x to be torch tensor of shape N x 3 x H x W with
        range -1 ... 1 if scale == 'symmetric' or range 0 ... 1 if range== 'asymmetric'
        :param x:
        :param scale:
        :return: torch tensor of shape N with CPBD scores
        """
        assert scale in ['symmetric', 'asymmetric']
        assert len(x.shape) in [3, 4]
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if scale == 'asymmetric':
            assert 0 <= x[0, 0, 0, 0] <= 1

        scores = []
        x_np = x.detach().cpu().permute(0, 2, 3, 1).numpy()
        if range == 'symmetric':
            x_np = x_np * .5 + .5
        x_np = (x_np * 255).astype(np.uint8)
        x_np[x_np < 0] = 0
        x_np[x_np > 255] = 255
        for x_ in x_np:
            #  0.299 * r + 0.587 * g + 0.114 * b
            x_gray = 0.299 * x_[:, :, 0] + 0.587 * x_[:, :, 1] + 0.114 * x_[:, :, 2]
            scores.append(cpbd.compute(x_gray))
        return torch.tensor(scores, device=self._device)


class EuclLmkDistance(Metric):
    """
    this module is based on https://github.com/1adrianb/face-alignment
    due to how tensors are handled in this 3rd party project internally, it is not differentiable
    """

    def __init__(self, device='cuda'):
        super().__init__(Metric.OPTIMUM_MIN, device)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                               flip_input=False,
                                               device=device)

    def __call__(self, x, gt, gt_landmarks=None, bbs=None, return_bbs=False):
        """
        assuming x being
        :param x: a predicted image tensor of shape N x 3 x H x W with range -1 ... 1
        :param gt: gt image tensor of shape N x 3 x H x W with range -1 ... 1.
                    Can be discarded if gt_landmarks are given.
        :param gt_landmarks: np array or torch tensor of gt landmarks of shape N x 68 x 2
        :param bbs: bounding boxes numpy array or torch tensor of shape  N x 5
        :return: L2 distance between gt_landmarks and predicted landmarks for x
        """
        if isinstance(gt_landmarks, torch.Tensor):
            gt_landmarks = gt_landmarks.detach().cpu().numpy()
        if isinstance(bbs, torch.Tensor):
            bbs = bbs.detach().cpu().numpy()

        if bbs is not None:
            assert bbs.shape[1] == 5
        if gt_landmarks is not None:
            assert gt_landmarks.shape[1:] == (68, 2)
        if bbs is not None and gt_landmarks is not None:
            assert len(bbs) == len(gt_landmarks)

        # preparing x for keypoint inference
        x_ = x.clone() * 127 + 127  # have to normalize to 0 ... 255 range for landmark detector
        gt_ = gt.clone() * 127 + 127  # have to normalize to 0 ... 255 range for landmark detector

        if bbs is None:
            bbs = self.fa.face_detector.detect_from_batch(gt_)
        else:
            # bringing numpy array bbs in same shape as predicted by face_detector:
            # list of lists of np arrays of length 5
            bbs = [[bb] for bb in bbs]

        # ignore samples where no face was detected
        keep_idcs = [i for i in range(len(bbs)) if len(bbs[i]) > 0]
        x_ = x_[torch.tensor(keep_idcs, dtype=torch.long, device=x_.device)]
        gt_ = gt_[torch.tensor(keep_idcs, dtype=torch.long, device=gt_.device)]
        bbs = [bbs[i] for i in keep_idcs]
        if gt_landmarks is not None:
            gt_landmarks = [gt_landmarks[i] for i in keep_idcs]

        # filter bbs with highest confidence for every image
        for i, det_bbs in enumerate(bbs):
            bbs[i] = [det_bbs[np.argmax(np.array(det_bbs)[:, -1])]]

        # determining gt landmarks
        if gt_landmarks is None:
            gt_landmarks = self.fa.get_landmarks_from_batch(gt_, detected_faces=bbs)
            gt_landmarks = np.array(gt_landmarks).reshape((gt_.shape[0], 68, 2))

        lmk = self.fa.get_landmarks_from_batch(x_, detected_faces=bbs)
        lmk = np.array(lmk).reshape((x_.shape[0], 68, 2))

        dist = torch.mean(
            torch.flatten(torch.from_numpy(np.sqrt(np.sum((lmk - gt_landmarks) ** 2, axis=-1))),
                          start_dim=1), dim=-1)
        return dist if not return_bbs else dist, np.array(bbs).reshape(-1, 5)


class LPIPS(Metric):
    def __init__(self, device='cuda'):
        super(LPIPS, self).__init__(Metric.OPTIMUM_MIN)
        self.net = lpips.LPIPS(pretrained=True, net="alex", eval_mode="True")

        self.net.eval()
        self.net.to(device)

    @torch.no_grad()
    def __call__(self, x, y):
        """
        x and y must be torch tensors of shape N x 3 x H x W with entries -1 ... 1
        :param x:
        :param y:
        :return:
        """
        return self.net.forward(x, y).reshape(x.shape[0])


class MS_SSIM(Metric):
    def __init__(self, device='cuda'):
        super().__init__(Metric.OPTIMUM_MAX, device)

    def __call__(self, x, y):
        # denormalizing input images
        x_ = x * .5 + .5
        y_ = y * .5 + .5
        return ms_ssim(x_, y_, data_range=1, size_average=False)


class PSNR(Metric):
    """Peak Signal to Noise Ratio
    img1 and img2 have range -1 ... 1"""

    def __init__(self, device='cuda'):
        super().__init__(Metric.OPTIMUM_MAX, device)

    def __call__(self, img1, img2):
        mse = torch.mean((torch.flatten((img1 - img2) ** 2, start_dim=1)), dim=-1)  # shape N
        return 20 * torch.log10(2. / torch.sqrt(mse))


from face_alignment.detection.sfd.sfd_detector import SFDDetector


class FaceBBxDetector:
    """
    this module is based on https://github.com/1adrianb/face-alignment
    due to how tensors are handled in this 3rd party project internally, it is not differentiable
    """

    def __init__(self, device='cuda'):
        self.detector = SFDDetector(device, filter_threshold=0.5)

    def __call__(self, x):
        """
        assuming x being
        :param x: a predicted image tensor of shape N x 3 x H x W with range -1 ... 1
        """

        # preparing x for bbx inference -> normalizing to
        x_ = x.clone() * 127 + 255  # have to normalize to 0 ... 255 range for landmark detector
        bbs = self.detector.detect_from_batch(x_)

        # filter bbs with highest confidence for every image
        for i, det_bbs in enumerate(bbs):
            if det_bbs:
                bbs[i] = [det_bbs[np.argmax(np.array(det_bbs)[:, -1])]]
            else:
                bbs[i] = [[np.nan] * 5]
        bbs = np.array(bbs).reshape(-1, 5)

        return bbs
