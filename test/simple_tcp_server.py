# save_as_simple_tcp_server.py
import socket

HOST = '127.0.0.1'  # 修改为你的 FastAPI 服务监听的 IP 地址
PORT = 8000         # 修改为你的 FastAPI 服务监听的端口

# 确保你的 FastAPI/Uvicorn 服务已停止，否则此脚本无法绑定到相同的端口

print(f"准备在 {HOST}:{PORT} 上监听原始 TCP 数据...")
print("请停止当前的 FastAPI/Uvicorn 服务，然后运行此脚本。")
print("脚本运行后，请从你的客户端发起有问题的请求。")
print("脚本会打印接收到的原始数据，然后等待下一个连接（或按 Ctrl+C 停止）。")

while True:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 允许地址重用
            s.bind((HOST, PORT))
            s.listen(1) # 只接受一个连接进行分析
            print(f"\n正在监听 {HOST}:{PORT}...")
            conn, addr = s.accept()
            with conn:
                print(f"客户端 {addr} 已连接。正在接收数据...")
                all_data = b''
                conn.settimeout(2.0) # 设置超时，防止无限期阻塞
                try:
                    while True:
                        chunk = conn.recv(4096) # 一次接收最多 4096 字节
                        if not chunk:
                            print("连接被客户端关闭。")
                            break
                        all_data += chunk
                        # 为了避免无限循环（如果客户端持续发送数据但不符合HTTP结束条件）
                        # 我们可以尝试简单地检测HTTP头的结束标志（双换行符）
                        # 或者在接收到一定量数据后停止。
                        # 对于调试 "Invalid HTTP request"，通常问题出在请求的开头部分。
                        if b"\r\n\r\n" in all_data: # 尝试检测HTTP头结束
                            print("检测到 HTTP 头部结束标志。")
                            break
                        if len(all_data) > 16 * 1024: # 限制最大读取量，防止内存溢出
                            print("已读取超过 16KB 数据，停止接收。")
                            break
                except socket.timeout:
                    print("接收数据超时。")
                except Exception as e:
                    print(f"接收数据时发生错误: {e}")

                print("\n----------- 客户端发送的原始数据 -----------")
                if all_data:
                    print("原始字节 (Hex):")
                    print(all_data.hex())
                    print("\n尝试以 UTF-8 解码 (仅供参考, 实际编码可能不同):")
                    try:
                        print(all_data.decode('utf-8', errors='replace'))
                    except Exception as e_decode:
                        print(f"UTF-8 解码失败: {e_decode}")
                else:
                    print("<未接收到数据>")
                print("----------------------------------------------")

                # 这个简单的服务器不会发送任何 HTTP 响应
                # 客户端可能会因此超时或显示错误，这是正常的
                # 我们的目标只是查看客户端发送的请求
                conn.close()
                print(f"与 {addr} 的连接已关闭。等待下一个连接...")

    except KeyboardInterrupt:
        print("\n服务器已停止。")
        break
    except Exception as e:
        print(f"服务器启动或运行时发生错误: {e}")
        print("请检查 HOST 和 PORT 设置是否正确，以及端口是否已被占用。")
        break
