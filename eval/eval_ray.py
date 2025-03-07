import ray
import time
# 初始化 Ray（默认会启动一个本地 Ray 集群）
ray.init()
# 基于远程装饰器，将普通函数变为分布式任务
@ray.remote
def compute_square(x):
    time.sleep(1)  # 模拟耗时计算
    return x * x
# 并行启动多个任务
numbers = [1, 2, 3, 4, 5]
future_results = [compute_square.remote(num) for num in numbers]
# 等待所有任务执行完毕，并收集结果
results = ray.get(future_results)
print(f"输入数字: {numbers}")
print(f"对应的平方值: {results}")
# 关闭 Ray
ray.shutdown()