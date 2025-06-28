#!/usr/bin/env python3
"""
自动化流程测试脚本
验证整个流水线是否正常工作
"""

import subprocess
import sys
import os
import json
import yaml

def run_command(cmd, description):
    """运行命令并检查结果"""
    print(f"\n🔄 {description}")
    print(f"执行: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            if result.stdout:
                print(f"输出: {result.stdout[:200]}...")
            return True
        else:
            print(f"❌ {description} 失败")
            print(f"错误: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} 异常: {e}")
        return False

def test_pipeline():
    """测试整个流水线"""
    print("🚀 开始测试自动化流水线")
    
    # 1. 检查数据
    success = run_command(
        "python scripts/data_pipeline.py --check-only",
        "数据管道检查"
    )
    if not success:
        return False
    
    # 2. 生成标签
    success = run_command(
        "python generate_micro_labels.py --horizon 120 --alpha 0.3 --mode maker --require-fill --output test_labels.parquet",
        "生成标签"
    )
    if not success:
        return False
    
    # 3. 训练模型
    success = run_command(
        "python train_micro_xgb.py --features test_labels.parquet --model-out test_model.bin --label-col label_h120_a0.3_maker",
        "训练模型"
    )
    if not success:
        return False
    
    # 4. 回测
    success = run_command(
        "python micro_backtest_maker.py --model test_model.bin --features test_labels.parquet --json-out test_results.json",
        "执行回测"
    )
    if not success:
        return False
    
    # 5. 选择最佳参数
    success = run_command(
        "python scripts/select_best.py test_results.json --metric sharpe --output test_best.yaml",
        "选择最佳参数"
    )
    if not success:
        return False
    
    # 6. 验证输出文件
    print("\n📋 验证输出文件:")
    
    files_to_check = [
        ("test_labels.parquet", "标签文件"),
        ("test_model.bin", "模型文件"),
        ("test_results.json", "回测结果"),
        ("test_best.yaml", "最佳参数")
    ]
    
    for filename, description in files_to_check:
        if os.path.exists(filename):
            print(f"✅ {description}: {filename}")
            if filename.endswith('.json'):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    print(f"   包含 {len(data)} 个结果")
            elif filename.endswith('.yaml'):
                with open(filename, 'r') as f:
                    data = yaml.safe_load(f)
                    print(f"   最佳Sharpe: {data.get('sharpe', 'N/A')}")
        else:
            print(f"❌ {description}: {filename} 不存在")
            return False
    
    print("\n🎉 自动化流水线测试完成！")
    return True

def cleanup():
    """清理测试文件"""
    test_files = [
        "test_labels.parquet",
        "test_model.bin", 
        "test_results.json",
        "test_best.yaml"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️  清理: {file}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        cleanup()
    else:
        success = test_pipeline()
        if success:
            print("\n✅ 所有测试通过，可以部署到 GitHub Actions")
        else:
            print("\n❌ 测试失败，请检查错误信息")
            sys.exit(1) 