import ray
import time

# 初始化 Ray，如果是在分布式集群环境中，可以传入 address 参数，例如 ray.init(address="auto")
ray.init()

# 使用 @ray.remote 装饰器定义一个远程任务，该任务声明需要使用 1 个 GPU
@ray.remote(num_gpus=1)
def train_model(task_id):
    """
    模拟在 GPU 上运行的任务，例如深度学习模型的训练任务。
    num_gpus=1 表示这个任务运行时需要预留一个 GPU。
    """
    # 获取 Ray 分配的 GPU id，可以用于确认任务使用了哪个 GPU
    gpu_ids = ray.get_gpu_ids()
    print(f"任务 {task_id} 正在使用 GPU: {gpu_ids}")
    
    # 模拟训练过程，此处用 sleep 来代表耗时操作
    time.sleep(5)
    
    print(f"任务 {task_id} 完成")
    return f"任务 {task_id} 的训练结果"

# 创建 8 个任务，每个任务需要分配 1 个 GPU
trained_tasks = [train_model.remote(i) for i in range(8)]

# 等待所有任务完成，并获取返回的结果
results = ray.get(trained_tasks)
print("所有任务的结果：", results)

# 关闭 Ray，释放资源
ray.shutdown()