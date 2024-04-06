import logging
logger = logging.getLogger('base')

# 创建DDPM模型的函数，用于在代码中的其他部分调用和使用该模型。
def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
