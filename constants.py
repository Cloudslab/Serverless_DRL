max_num_vms = 30
max_num_replicas = 80
max_total_pod_mem = 3000
min_pod_mem_req = 100
max_pod_cpu_req = 9200  # (1 core of type 2.3 GHz)
min_pod_cpu_req = 460  # ( 5% of a core)

max_vm_mem = 32000
max_vm_cpu = 73600

max_request_rate = 100
max_step_latency_perfn = 10
max_step_vmcost = 10

# desired util
pod_scale_cpu_util = 0.5
pod_scale_cpu_util_low = 0.2
pod_scale_cpu_util_high = 0.9
max_reschedule_tries = 6

# if a pod is not available, the wait time for a request to retry scheduling, in seconds
req_wait_time_to_schedule = 0.2

pod_creation_time = 3
pod_scheduling_time = 0
WL_duration = 300
step_interval = 9
scaling_start = 2
reward_interval = 4
max_steps = 6
reward_window_size = 3
state_latency_window = 1
num_episodes = 60
num_wls = 60

# Events tags
schedule_request = "SCHEDULE_REQ"
re_schedule_request = "RE_SCHEDULE_REQ"
finish_request = "REQ_COMPLETE"
create_new_pod = "CREATE_POD"
scale_pod = "SCALE_POD"
schedule_pod = "SCHEDULE_POD"
invoke_step_scaling = "STEP_SCALING"
calc_reward = "REWARD"

#Comparison algo parameters
knative_pod_concurrency = 3
kube_pod_cpu_threshold = 0.5
faas_capacity = 3
faas_rps_window = 1
faas_rps_threshold = 8
faas_cpu_threshold = 0.5
