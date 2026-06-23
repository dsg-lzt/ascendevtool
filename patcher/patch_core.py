import csv
import importlib
import logging
import os
import sys

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def perform_monkey_patch(unsupported_csv_path: str, local_ops_csv_path: str):
    """
    读取扫描出的不支持算子以及本地算子库配置，进行自动匹配与 Monkey Patch 动态挂载
    """
    if not os.path.exists(unsupported_csv_path):
        logging.warning(f"分析报告 {unsupported_csv_path} 不存在，跳过 Patch。")
        return
        
    if not os.path.exists(local_ops_csv_path):
        logging.warning(f"本地算子库配置文件 {local_ops_csv_path} 不存在，跳过 Patch。")
        return

    # 1. 解析目标不支持算子
    unsupported_ops = set()
    with open(unsupported_csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_name = row.get('OP', '').strip()
            if op_name:
                unsupported_ops.add(op_name)

    # 2. 解析本地算子库
    local_ops_mapping = {}
    with open(local_ops_csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig_op = row.get('OP', '').strip()
            mod = row.get('Local_Module', '').strip()
            func = row.get('Local_Func', '').strip()
            if orig_op and mod and func:
                local_ops_mapping[orig_op] = (mod, func)

    # 3. 匹配与动态替换
    patched_count = 0
    missing_ops = []

    for op in unsupported_ops:
        if op in local_ops_mapping:
            module_name, func_name = local_ops_mapping[op]
            try:
                # 动态加载我们自己写的本地算子库模块
                local_module = importlib.import_module(module_name)
                replacement_func = getattr(local_module, func_name)
                
                # 解析需要被劫持的原版 PyTorch API 路径
                # 例如 'torch.jit.annotate' -> torch_mod='torch.jit', torch_func='annotate'
                if '.' in op:
                    torch_mod_path, torch_func_name = op.rsplit('.', 1)
                    torch_mod = importlib.import_module(torch_mod_path)
                    # 关键动作：Monkey Patch 动态劫持
                    setattr(torch_mod, torch_func_name, replacement_func)
                    logging.info(f"成功挂载算子: {op} -> {module_name}.{func_name}")
                    patched_count += 1
                else:
                    logging.warning(f"暂不支持无命名空间的内置函数覆盖: {op}")
            except Exception as e:
                logging.error(f"替换算子 {op} 失败: {e}")
        else:
            missing_ops.append(op)

    # 4. 打印匹配报告
    logging.info(f"Patch 完成，共补齐 {patched_count} 个算子。")
    if missing_ops:
        logging.warning(f"以下算子在本地算子库中缺失，请交由 OpBuilder 进行 Ascend C 开发:")
        for mop in missing_ops:
            logging.warning(f"  - {mop}")

# 在执行 import patch_core 的时候自动生效
if __name__ != "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_csv = os.path.join(BASE_DIR, 'scanner', 'reports', 'unsupported_api.csv')
    local_lib_csv = os.path.join(BASE_DIR, 'patcher', 'local_op_lib', 'local_ops.csv')
    
    # 确保当前项目根目录在系统路径中，以防模块找不到
    sys.path.insert(0, BASE_DIR)
    
    perform_monkey_patch(report_csv, local_lib_csv)
