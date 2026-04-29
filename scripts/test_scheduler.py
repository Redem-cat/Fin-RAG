"""
Scheduler 诊断脚本
用法: python scripts/test_scheduler.py (需确保 Neo4j 和 Ollama 至少有一个运行)
"""
import sys, os, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

SEP = "=" * 60


def test():
    print(f"\n{SEP}")
    print(f"  KG Scheduler 诊断工具")
    print(f"{SEP}\n")

    # ===== 阶段1: 导入 =====
    print("[1] 模块导入")
    try:
        from src.kg_crawler.scheduler import setup_default_tasks, get_kg_scheduler, KGScheduler
        print(f"  ✅ scheduler 模块导入成功")
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        traceback.print_exc()
        return
    except Exception as e:
        print(f"  ❌ 未知错误: {e}")
        traceback.print_exc()
        return

    # ===== 阶段2: 创建调度器实例 =====
    print(f"\n[2] 调度器实例化")
    try:
        sched = get_kg_scheduler()
        print(f"  ✅ 实例创建成功, 类型: {type(sched).__name__}")
    except Exception as e:
        print(f"  ❌ 实例化失败: {e}")
        traceback.print_exc()
        return

    # ===== 阶段3: 注册任务（不触发依赖）=====
    print(f"\n[3] 任务注册 (setup_default_tasks)")
    try:
        setup_default_tasks()
        status = sched.get_status()
        tasks = status.get("tasks", {})
        print(f"  ✅ 注册了 {len(tasks)} 个任务:")
        for name, info in tasks.items():
            print(f"     - {name} ({info['schedule_type']})")
    except Exception as e:
        print(f"  ❌ 任务注册失败: {e}")
        traceback.print_exc()
        return

    # ===== 阶段4: 依赖健康检查 =====
    print(f"\n[4] 依赖健康检查")
    try:
        health = sched.check_dependencies()
        for name, info in health.items():
            icon = "OK" if info["ok"] else "FAIL"
            detail = info.get("detail", "")
            if len(detail) > 60:
                detail = detail[:57] + "..."
            print(f"  [{icon}] {name}: {detail}")
    except Exception as e:
        print(f"  ⚠️ 依赖检查异常: {e}")

    # ===== 阶段5: 尝试启动（非阻塞）=====
    print(f"\n[5] 启动调度器 (blocking=False)")
    try:
        sched.start(blocking=False)
        import time
        time.sleep(1)
        final_status = sched.get_status()
        print(f"  ✅ 调度器已启动!")
        print(f"     running={final_status['running']}, tasks={final_status['tasks_count']}")

        # 停止
        sched.stop()
        print(f"     已停止测试用调度器")
    except Exception as e:
        print(f"  ❌ 启动失败: {e}")
        traceback.print_exc()

    print(f"\n{SEP}")


if __name__ == "__main__":
    test()
