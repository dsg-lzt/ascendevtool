#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPS_SRC_DIR = os.path.join(BASE_DIR, "ops_src")
OPLIB_PACKAGES_DIR = os.path.abspath(os.path.join(BASE_DIR, "../oplib/custom_opp_packages"))

def run_cmd(cmd, cwd=None):
    print(f"\n[RUN] >>> {cmd} (cwd: {cwd or os.getcwd()})")
    ret = subprocess.run(cmd, shell=True, cwd=cwd)
    if ret.returncode != 0:
        print(f"[ERROR] 执行失败: {cmd}")
        sys.exit(ret.returncode)

def generate(op_name, json_path):
    print(f"========== 1. GENERATE: {op_name} ==========")
    target_dir = os.path.join(OPS_SRC_DIR, f"{op_name}Sample", "FrameworkLaunch")
    os.makedirs(target_dir, exist_ok=True)
    json_path = os.path.abspath(json_path)
    
    # 绕过 msopgen 对于中文路径和文件权限的安全限制
    import shutil
    import json
    
    tmp_workspace = os.path.expanduser("~/ascend_gen_workspace")
    os.makedirs(tmp_workspace, exist_ok=True)
    os.chmod(tmp_workspace, 0o755)  # 严格要求非 group-writable
    
    tmp_json = os.path.join(tmp_workspace, f"{op_name}.json")
    run_cmd(f"cp '{json_path}' {tmp_json}")
    os.chmod(tmp_json, 0o644)       # 严格要求非 group-writable

    # Parse JSON to find all operators
    with open(tmp_json, 'r') as f:
        op_data = json.load(f)
    
    if isinstance(op_data, list):
        op_names = [item["op"] for item in op_data if "op" in item]
    else:
        op_names = [op_data.get("op")] if op_data.get("op") else []

    if not op_names:
        print("[ERROR] 无法在 JSON 中找到算子名称 ('op' 字段)")
        sys.exit(1)

    print(f"[INFO] 识别到 {len(op_names)} 个算子：{', '.join(op_names)}")

    soc_ver = os.environ.get("SOC_VERSION", "Ascend310P")
    
    for i, single_op_name in enumerate(op_names):
        mode = "0" if i == 0 else "1"
        print(f"\n---> [{i+1}/{len(op_names)}] 正在生成/添加算子：{single_op_name}")
        # cmd = f"msopgen gen -i {op_name}.json -c ai_core-{soc_ver} -lan cpp -out {op_name} -op {single_op_name} -m {mode}"
        cmd = f"msopgen gen -i {op_name}.json -c ai_core-{soc_ver} -lan cpp -out {op_name} -op {single_op_name}"
        run_cmd(cmd, cwd=tmp_workspace)
    
    # 自动修改 CMakePresets.json 以防所有的 vendor_name 都叫 customize
    preset_file = os.path.join(tmp_workspace, op_name, "CMakePresets.json")
    if os.path.exists(preset_file):
        with open(preset_file, 'r') as f:
            preset_data = json.load(f)
        for preset in preset_data.get("configurePresets", []):
            if "cacheVariables" in preset and "vendor_name" in preset["cacheVariables"]:
                preset["cacheVariables"]["vendor_name"]["value"] = op_name.lower()
        with open(preset_file, 'w') as f:
            json.dump(preset_data, f, indent=4)
        print(f"[INFO] 自动替换了 vendor_name 为：{op_name.lower()}")
    
    # 拷贝生成产物回到带有中文路径的真实工程目录
    generated_dir = os.path.join(tmp_workspace, op_name)
    run_cmd(f"cp -r {generated_dir} {target_dir}/")
    run_cmd(f"cp {tmp_json} {target_dir}/")
    
    # 物理清理
    run_cmd(f"rm -rf {tmp_workspace}/*")

    print(f"[SUCCESS] Ascend C 工程已生成：{os.path.join(target_dir, op_name)}")
    print(f"请前去修改 op_kernel 和 op_host 补充算子实现！")

def build(op_name):
    print(f"========== 2. BUILD: {op_name} ==========")
    build_dir = os.path.join(OPS_SRC_DIR, f"{op_name}Sample", "FrameworkLaunch", op_name)
    if not os.path.exists(build_dir):
        print(f"[ERROR] 找不到工程目录: {build_dir}！请先生成。")
        sys.exit(1)
        
    cmd = "bash build.sh"
    run_cmd(cmd, cwd=build_dir)
    print(f"[SUCCESS] 编译完成，OPP 包已在 {build_dir}/build_out 目录下！")

def verify(op_name):
    print(f"========== 3. VERIFY: {op_name} ==========")
    invoke_dir = os.path.join(OPS_SRC_DIR, f"{op_name}Sample", "FrameworkLaunch", "AclNNInvocation")
    if not os.path.exists(invoke_dir):
        print(f"[WARN] 未找到 AclNNInvocation 目录: {invoke_dir}，跳过验证步骤。")
        return
        
    run_cmd("bash run.sh", cwd=invoke_dir)
    print(f"[SUCCESS] 算子精度与运行验证通过！")

def install(op_name):
    print(f"========== 4. INSTALL (归库): {op_name} ==========")
    build_out_dir = os.path.join(OPS_SRC_DIR, f"{op_name}Sample", "FrameworkLaunch", op_name, "build_out")
    
    # 查找 .run 安装包 (类似 custom_opp_ubuntu_aarch64.run)
    run_pkgs = glob.glob(os.path.join(build_out_dir, "*.run"))
    if not run_pkgs:
        print(f"[ERROR] 找不到安装包 {build_out_dir}/*.run！请确认编译成功。")
        sys.exit(1)
        
    pkg = run_pkgs[0]
    pkg_name = os.path.basename(pkg)
    
    # 1. 归档到本地 oplib/custom_opp_packages/
    os.makedirs(OPLIB_PACKAGES_DIR, exist_ok=True)
    target_pkg = os.path.join(OPLIB_PACKAGES_DIR, pkg_name)
    run_cmd(f"cp {pkg} {target_pkg}")
    
    # 2. 自动安装到 CANN 环境
    print("[INFO] 正将算子安装到当前 CANN 环境变量 (ASCEND_OPP_PATH)...")
    run_cmd(f"{target_pkg} --quiet")
    print(f"[SUCCESS] 算子 {op_name} 已安装并归档至: {target_pkg}")

def main():
    parser = argparse.ArgumentParser(description="Ascend C OpBuilder 生命周期管理工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    p_gen = subparsers.add_parser("generate", help="依靠 JSON 描述快速生成 Ascend C 工程")
    p_gen.add_argument("op_name", help="算子名 (如 BallQuery)")
    p_gen.add_argument("--json", required=True, help="算子定义 JSON 文件路径")
    
    p_build = subparsers.add_parser("build", help="编译 Ascend C 算子生成 run 包")
    p_build.add_argument("op_name", help="算子名 (如 BallQuery)")
    
    p_verify = subparsers.add_parser("verify", help="单算子 AclNN 执行验证")
    p_verify.add_argument("op_name", help="算子名 (如 BallQuery)")

    p_install = subparsers.add_parser("install", help="部署归入本地库 (oplib)")
    p_install.add_argument("op_name", help="算子名 (如 BallQuery)")

    p_all = subparsers.add_parser("pipeline", help="一件执行 build -> verify -> install 闭环")
    p_all.add_argument("op_name", help="算子名 (如 BallQuery)")

    args = parser.parse_args()
    
    if args.command == "generate":
        generate(args.op_name, args.json)
    elif args.command == "build":
        build(args.op_name)
    elif args.command == "verify":
        verify(args.op_name)
    elif args.command == "install":
        install(args.op_name)
    elif args.command == "pipeline":
        build(args.op_name)
        verify(args.op_name)
        install(args.op_name)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
