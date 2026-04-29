"""
锐思(RESSET) API 诊断脚本
用法: python scripts/test_resset.py
"""

import sys
import os
import socket
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEPARATOR = "=" * 60


def check_python():
    print(f"\n[1] Python 环境")
    print(f"  Python: {sys.version}")
    print(f"  平台: {sys.platform}")


def check_package():
    """检查 resset 包是否安装"""
    print(f"\n[2] 包检查")
    try:
        import resset
        print(f"  ✅ resset 已安装, 路径: {resset.__file__}")
        try:
            from resset.report import reportdata
            print(f"  ✅ reportdata 模块可用")
            print(f"  可用属性: {[a for a in dir(reportdata) if not a.startswith('_')]}")
            return True
        except ImportError as e:
            print(f"  ❌ reportdata 导入失败: {e}")
            return False
    except ImportError as e:
        print(f"  ❌ resset 未安装: {e}")
        print(f"  安装命令: pip install https://rtas.resset.com/txtPath/resset-0.9.8-py3-none-any.whl")
        return False
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        traceback.print_exc()
        return False


def check_network():
    """检查网络连通性"""
    print(f"\n[3] 网络连通性")
    hosts = [
        ("rtas.resset.com", 443, "锐思API服务器 (HTTPS)"),
        ("www.resset.com", 443, "锐思官网 (HTTPS)"),
    ]

    results = []
    for host, port, desc in hosts:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            print(f"  ✅ {desc} - {host}:{port} 通")
            results.append(True)
        except socket.timeout:
            print(f"  ❌ {desc} - {host}:{port} 超时 (可能需要校园网/机构IP)")
            results.append(False)
        except socket.gaierror:
            print(f"  ❌ {desc} - DNS解析失败 ({host})")
            results.append(False)
        except Exception as e:
            print(f"  ⚠️ {desc} - {host}:{port} 错误: {e}")
            results.append(False)

    return all(results)


def check_login():
    """测试登录"""
    print(f"\n[4] 登录测试")

    try:
        from resset.report import reportdata
    except ImportError:
        print("  跳过: resset 未安装")
        return False

    username = "sysu"
    password = "sysu"
    print(f"  用户名: {username}")
    print(f"  密码: {'*' * len(password)}")

    try:
        # 尝试登录，打印完整调用栈信息
        print(f"  调用: reportdata.ressetLogin('{username}', '...')...")
        login_id = reportdata.ressetLogin(username, password)
        print(f"  ✅ 登录成功! login_id = {login_id}")

        # 尝试获取权限
        try:
            perm = reportdata.get_Permission(login_id)
            print(f"  权限信息: {perm[:200] if perm else 'None'}")
        except Exception as pe:
            print(f"  权限查询异常: {pe}")

        return True

    except ImportError as ie:
        print(f"  ❌ 导入错误: {ie}")
        return False
    except Exception as e:
        print(f"  ❌ 登录失败: {type(e).__name__}: {e}")
        print(f"\n  --- 完整错误堆栈 ---")
        traceback.print_exc()
        print(f"  --- 结束 ---")

        # 常见原因提示
        err_str = str(e).lower()
        if "timeout" in err_str or "timed out" in err_str:
            print(f"\n  💡 提示: 连接超时，请检查:")
            print(f"     - 是否连接校园网/机构IP")
            print(f"     - 防火墙是否拦截出站连接")
        elif "auth" in err_str or "login" in err_str or "password" in err_str or "credential" in err_str:
            print(f"\n  💡 提示: 认证失败，请检查:")
            print(f"     - 账号密码是否正确")
            print(f"     - 账号是否过期或额度耗尽")
        elif "connection" in err_str or "network" in err_str or "refused" in err_str:
            print(f"\n  💡 提示: 网络连接问题，请检查:")
            print(f"     - 锐思服务是否在维护")
            print(f"     - VPN/代理设置")
        elif "ssl" in err_str or "certificate" in err_str:
            print(f"\n  💡 提示: SSL证书问题，尝试:")
            print(f"     - 更新 Python SSL 证书: pip install --upgrade certifi")

        return False


def check_simple_query():
    """简单数据获取测试"""
    print(f"\n[5] 数据获取测试 (仅当登录成功时)")

    try:
        from src.resset_data import get_resset_connection
        conn = get_resset_connection()

        if not conn.is_available:
            print("  跳过: 登录未成功")
            return

        print("  尝试获取政府工作报告 (2023年)...")
        data = conn.get_content_data("100100", "part", "政府工作报告", "2023")
        if data:
            print(f"  ✅ 获取到 {len(data)} 条数据")
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                print(f"  第一条数据字段: {list(first.keys()) if isinstance(first, dict) else type(first)}")
        else:
            print(f"  ⚠️ 无数据返回（可能是正常情况）")

    except Exception as e:
        print(f"  ❌ 数据获取异常: {e}")


if __name__ == "__main__":
    print(SEPARATOR)
    print("  锐思(RESSET) API 诊断工具")
    print(SEPARATOR)

    check_python()
    ok_pkg = check_package()
    ok_net = check_network()

    if ok_pkg and ok_net:
        ok_login = check_login()
        if ok_login:
            check_simple_query()
    elif not ok_pkg:
        print(f"\n{'='*60}")
        print(f"  结论: resset 包未安装，无法测试登录")
        print(f"  请先执行安装命令")
        print(f"{'='*60}")
    elif not ok_net:
        print(f"\n{'='*60}")
        print(f"  结论: 网络不通，无法连接锐思服务器")
        print(f"  请确认当前网络环境（需要校园网/机构IP）")
        print(f"{'='*60}")

    print(f"\n{SEPARATOR}")
    print("  诊断完成")
    print(SEPARATOR)
