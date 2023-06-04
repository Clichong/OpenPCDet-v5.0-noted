from pathlib import Path

import yaml
from easydict import EasyDict

# 作用：将字典信息递归的全部打印来
def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):      # 字典的val是EasyDict进行递归打印
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


# 作用：用额外的列表对cfg配置文件进行参数的更改
def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)   # 将布尔变量的字符串改为布尔变量 eg：'True' -> True
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


# 作用：加载yaml文件具体执行遍历的函数，与log_config_to_file类似的写法，递归遍历yaml文件的每个字典
def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):   # 非字典模式直接赋值
            config[key] = val
            continue
        if key not in config:   # 字典模式则创建空字典进行递归赋值
            config[key] = EasyDict()
        merge_new_config(config[key], val)  # 递归

    return config


# 作用：加载yaml文件
def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)  # 具体的遍历函数

    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()     # 这里的resolve是返回文件的绝对路径
cfg.LOCAL_RANK = 0
