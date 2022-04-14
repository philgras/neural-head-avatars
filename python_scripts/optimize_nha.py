from nha.optimization.train_pl_module import train_pl_module
from nha.util.log import get_logger
from nha.data.real import RealDataModule
from nha.models.nha_optimizer import NHAOptimizer

logger = get_logger("nha", root=True)

if __name__ == "__main__":
    train_pl_module(NHAOptimizer, RealDataModule)
