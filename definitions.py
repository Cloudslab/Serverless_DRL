class VM:
    def __init__(self, vm_type, vm_id, cpu, mem, cpu_u, mem_u, cpu_allo, mem_allo, price, nw_bandwidth, io_bandwidth, run_pod_list, term_pod_list):
        self.type = vm_type
        self.id = vm_id
        self.cpu = cpu
        self.ram = mem
        self.cpu_used = cpu_u
        self.ram_used = mem_u
        self.cpu_allocated = cpu_allo
        self.mem_allocated = mem_allo
        self.price = price
        self.bandwidth = nw_bandwidth
        self.diskio_bw = io_bandwidth
        self.running_pod_list = run_pod_list
        self.term_pod_list = term_pod_list


class POD:
    def __init__(self, id, vm_allo, pod_type, start_t, term_t, cpu, r_mips,  mem, cpu_u, ram_u, rps, cur_replicas, max_replicas, running_requests, run_list, com_list):
        self.id = id
        self.allocated_vm = vm_allo
        self.type = pod_type
        self.start_time = start_t
        self.term_time = term_t
        self.cpu_req = cpu
        self.req_mips = r_mips
        self.ram_req = mem
        self.req_per_sec = rps
        self.cpu_util = cpu_u
        self.ram_util = ram_u
        self.num_current_replicas = cur_replicas
        self.num_max_replicas = max_replicas
        self.num_inflight_requests = running_requests
        self.running_req_list = run_list
        self.completed_req_list = com_list



class REQUEST:
    def __init__(self, id, num, type, pod_allo, arr_t, st_t, fin_t, rate, tries, stat, fn2_name, status):
        self.id = id
        self.chain = num
        self.type = type
        self.allocated_pod = pod_allo
        self.arrival_time = arr_t
        self.start_time = st_t
        self.finish_time = fin_t
        self.arrival_rate = rate
        self.reschedule_tries = tries
        self.status = stat
        self.fn2 = fn2_name
        self.done = status


class EVENT:
    def __init__(self, time, ev_name, obj):
        self.received_time = time
        self.event_name = ev_name
        self.entity_object = obj
