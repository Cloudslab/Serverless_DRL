import logging
import math
import re
import threading
import uuid
from re import match
import xlwt
from xlwt import Workbook
from threading import Thread
import xlrd
import constants
import cluster
import definitions as defs
import numpy as np
from sys import maxsize

vm_features = cluster.vm_features
vm_lock = threading.Lock()
history_lock = threading.Lock()
mode = ""


class Worker:
    def __init__(self, worker, episode):
        self.sorted_request_history_per_window = {}
        self.pod_object_dict_by_type = {}
        self.pending_pods = {}
        self.required_pod_count = {}
        self.pod_counter = {}
        self.fn_request_rate = {}
        self.pods = []
        self.sorted_events = []
        self.dropped_requests = []
        self.clock = 0
        self.simulation_running = True
        self.vm_up_time_dict = {}
        self.vm_up_time_cost_prev = 0
        self.episodic_reward = 0
        self.function_latency = 0
        self.fn_failures = 0
        self.ver_cpu_action_total = 0
        self.ver_mem_action_total = 0
        self.hor_action_total = 0
        self.Episodic_failure_rate = 0
        self.Episodic_latency = 0
        self.total_vm_cost_diff = 0
        self.pod_id = 1
        self.episode_no = episode
        self.worker_id = worker
        self.episodic_loss = 0

        self.vms = cluster.init_vms()

        self.serv_env_state_init = cluster.gen_serv_env_state_init()
        self.serv_env_state_min = cluster.gen_serv_env_state_min()
        self.serv_env_state_max = cluster.gen_serv_env_state_max()

        logging.info(
            "ServEnv_Base to be initialized for worker {} episode: {}".format(worker, episode))

        self.fn_features = cluster.init_fn_features()

        self.wb = Workbook()
        self.req_wl = self.wb.add_sheet('Results')
        self.req_wl.write(0, 0, 'Fn ID')
        self.req_wl.write(0, 1, 'Fn type')
        self.req_wl.write(0, 2, 'Allocated pod')
        self.req_wl.write(0, 3, 'Arrival time')
        self.req_wl.write(0, 4, 'Start/Dropped time')
        self.req_wl.write(0, 5, 'Finish time')
        self.req_wl.write(0, 6, 'Result')

        self.pod_data_sheet = self.wb.add_sheet('Pods')
        self.pod_data_sheet.write(0, 0, 'Pod ID')
        self.pod_data_sheet.write(0, 1, 'Pod type')
        self.pod_data_sheet.write(0, 2, 'Start time')
        self.pod_data_sheet.write(0, 3, 'Finish time')

        self.results_sheet_row_counter = 1
        self.pod_sheet_row_counter = 1

        self.fn_types = []
        self.fn_iterator = -1

        self.init_workload()
        # self.init_workload_test()

    def sorter_events(self, item):
        time = float(item.received_time)
        return time

    def pod_sorter(self, item):
        util = float(item.cpu_util)
        return util

    def pod_creator(self, fn_type):
        global vm_features
        run_req_list = []
        term_req_list = []
        running_pod_list = {}
        term_pod_list = []
        vm = defs.VM(0, len(self.vms), int(vm_features['vm0_cpu_total_speed']), int(vm_features['vm0_mem']),
                     int(vm_features['vm0_cpu_used']), int(vm_features['vm0_mem_used']),
                     int(vm_features['vm0_cpu_allocated']),
                     int(vm_features['vm0_mem_allocated']), int(vm_features['vm0_price']),
                     int(vm_features['vm0_nw_bandwidth']),
                     int(vm_features['vm0_diskio_bandwidth']), running_pod_list, term_pod_list)
        self.pods.append(
            defs.POD(self.pod_id, vm, self.fn_features[str(fn_type) + "_name"], 0, 0,
                     int(self.fn_features[str(fn_type) + "_pod_cpu_req"]),
                     int(self.fn_features[str(fn_type) + "_req_MIPS"]),
                     int(self.fn_features[str(fn_type) + "_pod_ram_req"]),
                     int(self.fn_features[str(fn_type) + "_pod_cpu_util"]),
                     int(self.fn_features[str(fn_type) + "_pod_ram_util"]),
                     int(self.fn_features[str(fn_type) + "_req_per_sec"]),
                     int(self.fn_features[str(fn_type) + "_num_current_replicas"]),
                     int(self.fn_features[str(fn_type) + "_num_max_replicas"]),
                     int(self.fn_features[str(fn_type) + "_inflight_requests"]), run_req_list, term_req_list))

        if self.pod_id == 1:
            print("POD 1")

        # logging.info(
        #     "CLOCK: {}  Pod id: {} Type: {} worker: {} created. Req sent to schedule ".format(self.clock, self.pod_id,
        #                                                                                       fn_type, self.worker_id))

        self.pod_id += 1

        self.sorted_events.append(defs.EVENT(self.clock + constants.pod_scheduling_time, constants.schedule_pod,
                                             self.pods[len(self.pods) - 1]))
        self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)

    def pod_scaler(self):
        pod_info = {}

        for pod in self.pods:
            completed_count = 0
            if pod.type in pod_info:
                if pod in self.pod_object_dict_by_type[pod.type]:
                    # pod.cpu_util is in no of cores (not %)
                    pod_info[pod.type]['pod_cpu_util_total'] += pod.cpu_util
                    pod_info[pod.type]['pod_count'] += 1
                    pod_info[pod.type]['pod_inflight_request_count'] += pod.num_inflight_requests
                    for req in reversed(pod.completed_req_list):
                        if req.finish_time > self.clock - constants.faas_rps_window:
                            if req.status != "Dropped":
                                completed_count += 1
                    pod_info[pod.type]['pod_completed_requests'] += completed_count

            else:
                if pod.type in self.pod_object_dict_by_type:
                    if pod in self.pod_object_dict_by_type[pod.type]:
                        pod_info[pod.type] = {}
                        pod_info[pod.type]['pod_cpu_util_total'] = pod.cpu_util
                        pod_info[pod.type]['pod_count'] = 1
                        pod_info[pod.type]['pod_inflight_request_count'] = pod.num_inflight_requests
                        for req in reversed(pod.completed_req_list):
                            if req.finish_time > self.clock - constants.faas_rps_window:
                                if req.status != "Dropped":
                                    completed_count += 1
                        pod_info[pod.type]['pod_completed_requests'] = completed_count
        # logging.info(
        #     "CLOCK: {} worker: {} Now in pod scaler. now running pod count: {} ".format(self.clock, self.worker_id,
        #                                                                                 pod_info['pod_count']))
        for pod_type, pod_data in pod_info.items():
            # ***********COMPARISON ALGOS - Also change action to no changes
            if mode == 'knative':
                desired_replicas = math.ceil(pod_data['pod_count'] * ((pod_data['pod_inflight_request_count'] /
                                                                       pod_data[
                                                                           'pod_count']) / constants.knative_pod_concurrency))
            elif mode == 'kube':
                desired_replicas = math.ceil(pod_data['pod_count'] * (pod_data['pod_cpu_util_total'] / int(
                    self.fn_features[str(pod_type) + "_pod_cpu_req"])) / pod_data[
                                                 'pod_count'] / constants.kube_pod_cpu_threshold)
            elif mode == 'faas':
                if self.fn_features[str(pod_type) + "_req_exec_time"] > 2:
                    desired_replicas = math.ceil(pod_data['pod_count'] * ((pod_data['pod_inflight_request_count'] /
                                                                           pod_data[
                                                                               'pod_count']) / constants.faas_capacity))
                elif self.fn_request_rate[pod_type] > 20:
                    desired_replicas = math.ceil(pod_data['pod_count'] * ((pod_data['pod_completed_requests'] /
                                                                           pod_data[
                                                                               'pod_count']) / constants.faas_rps_threshold))
                    print("pod count: {} total completed requests: {}".format(pod_data['pod_count'],
                                                                              pod_data['pod_completed_requests']))
                else:
                    desired_replicas = math.ceil(pod_data['pod_count'] * (pod_data['pod_cpu_util_total'] / int(
                        self.fn_features[str(pod_type) + "_pod_cpu_req"])) / pod_data[
                                                     'pod_count'] / constants.faas_cpu_threshold)
            # ************************************************
            else:
                logging.info(
                    "CLOCK: {}  now running pod count: {} Pod type: {} threshold: {} cpu util: {} cpu req: {}".format(
                        self.clock, pod_data['pod_count'], pod_type, self.fn_features[
                            str(pod_type) + "_scale_cpu_threshold"], pod_data['pod_cpu_util_total'], int(
                            self.fn_features[str(pod_type) + "_pod_cpu_req"])))
                desired_replicas = math.ceil(pod_data['pod_count'] * (pod_data['pod_cpu_util_total'] / int(
                    self.fn_features[str(pod_type) + "_pod_cpu_req"])) / pod_data['pod_count'] / self.fn_features[
                                                 str(pod_type) + "_scale_cpu_threshold"])
                logging.info("CLOCK: {} Desired replicas: {} now running pod count: {} Pod type: {}".format(self.clock,
                                                                                                            desired_replicas,
                                                                                                            pod_data[
                                                                                                                'pod_count'],
                                                                                                            pod_type))
            if desired_replicas > 0:
                new_pod_count = min(desired_replicas, int(constants.max_num_replicas))
            else:
                new_pod_count = 1

            self.required_pod_count[pod_type] = new_pod_count
            if pod_type in self.pending_pods:
                pending = self.pending_pods[pod_type]
            else:
                pending = 0
            # logging.info("CLOCK: {} worker: {} No of pending pods of this type: {}".format(self.clock, self.worker_id, pending))
            if new_pod_count > pod_data['pod_count'] + pending:
                new_pods_to_create = new_pod_count - pod_data['pod_count'] - pending
                logging.info(
                    "CLOCK: {} worker: {} fn: {} now running pod count: {} pending pods: {} Scaling up, new pods to create: {}".format(
                        self.clock, self.worker_id, pod_type, pod_data['pod_count'], self.pending_pods[pod_type],
                        new_pods_to_create))
                for x in range(new_pods_to_create):
                    self.sorted_events.append(
                        defs.EVENT(self.clock + constants.pod_creation_time, constants.scale_pod, pod_type))
                    self.pending_pods[pod_type] += 1

            # scaling down of pods if pod has no running requests
            if new_pod_count < pod_data['pod_count'] + pending:
                pods_to_scale_down = pending + pod_data['pod_count'] - new_pod_count
                logging.info(
                    "CLOCK: {} worker: {} Scaling down, num pods to remove: {}".format(self.clock, self.worker_id,
                                                                                       pods_to_scale_down))
                pod_list_length = len(self.pod_object_dict_by_type[pod_type])
                removed_pods = 0
                pods_to_remove = []
                for i in range(pod_list_length):
                    pod = self.pod_object_dict_by_type[pod_type][i]
                    if pod.num_inflight_requests == 0:
                        # with vm_lock:
                        # logging.info(
                        #     "CLOCK: {} worker: {} Pod Id: {} to be removed, Allocated VM id: {}, vm running pod list: {}".format(self.clock, self.worker_id, pod.id,
                        #                                                                                     pod.allocated_vm.id,
                        #                                                                                     pod.allocated_vm.running_pod_list))
                        # print(PODS)
                        self.pods.remove(pod)
                        pods_to_remove.append(pod)
                        pod.allocated_vm.running_pod_list[pod.type].remove(pod)

                        if len(pod.allocated_vm.running_pod_list[pod.type]) == 0:
                            pod.allocated_vm.running_pod_list.pop(pod.type)

                        if len(pod.allocated_vm.running_pod_list) == 0:
                            self.vm_up_time_dict[pod.allocated_vm]['status'] = "OFF"
                            self.vm_up_time_dict[pod.allocated_vm]['total_time'] += self.clock - \
                                                                                    self.vm_up_time_dict[
                                                                                        pod.allocated_vm][
                                                                                        'time_now']
                            logging.info(
                                "clock: {} Worker: {} vm {} up time so far: {} latest segment: {}".format(self.clock,
                                                                                                          self.worker_id,
                                                                                                          pod.allocated_vm.id,
                                                                                                          self.vm_up_time_dict[
                                                                                                              pod.allocated_vm][
                                                                                                              'total_time'],
                                                                                                          self.clock - \
                                                                                                          self.vm_up_time_dict[
                                                                                                              pod.allocated_vm][
                                                                                                              'time_now']))

                            self.vm_up_time_dict[pod.allocated_vm]['time_now'] = self.clock

                        pod.allocated_vm.cpu_allocated -= pod.cpu_req
                        pod.allocated_vm.mem_allocated -= pod.ram_req
                        pod.term_time = self.clock
                        removed_pods += 1

                        self.pod_data_sheet.write(self.pod_sheet_row_counter, 0, pod.id)
                        self.pod_data_sheet.write(self.pod_sheet_row_counter, 1, pod.type)
                        self.pod_data_sheet.write(self.pod_sheet_row_counter, 2, pod.start_time)
                        self.pod_data_sheet.write(self.pod_sheet_row_counter, 3, pod.term_time)
                        self.pod_sheet_row_counter += 1

                        if removed_pods == pods_to_scale_down:
                            break
                self.wb.save(
                    "results/" + str(self.worker_id) + "/" + "Results_Episode_" + str(self.episode_no) + ".xls")

                for p in pods_to_remove:
                    logging.info(
                        "Clock {} worker: {} Pod {} of type {} removed from list".format(self.clock, self.worker_id,
                                                                                         p.id,
                                                                                         p.type))
                    self.pod_object_dict_by_type[p.type].remove(p)

        self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)

        for pod in self.pods:
            if pod.type in self.pod_object_dict_by_type:
                pod.num_current_replicas = len(self.pod_object_dict_by_type[pod.type])

    def pod_scheduler(self, pod):
        vm_allocated = False
        if self.episode_no == 4:
            print("debug at pod scheduler ")
        # with vm_lock:
        for vm in self.vms:
            # print("CPU: " + str(vm.cpu) + " CPU allocated: " + str(vm.cpu_allocated))
            # print("ram: " + str(vm.ram) + " ram allocated: " + str(vm.mem_allocated))
            if ((vm.cpu - vm.cpu_allocated) >= pod.cpu_req) & ((vm.ram - vm.mem_allocated) >= pod.ram_req):
                pod.allocated_vm = vm
                logging.info(
                    "worker: {} Pod {} type: {} is allocated to VM {}".format(self.worker_id, pod.id, pod.type, vm.id))
                vm.cpu_allocated += pod.cpu_req
                vm.mem_allocated += pod.ram_req
                if vm in self.vm_up_time_dict:
                    if self.vm_up_time_dict[vm]['status'] != "ON":
                        # logging.info(
                        #     "VM {} is already in ON status so not making changes to dict".format(vm.id))
                        # else:
                        self.vm_up_time_dict[vm]['status'] = "ON"
                        self.vm_up_time_dict[vm]['time_now'] = self.clock
                else:
                    self.vm_up_time_dict[vm] = {}
                    self.vm_up_time_dict[vm]['status'] = "ON"
                    self.vm_up_time_dict[vm]['time_now'] = self.clock
                    self.vm_up_time_dict[vm]['total_time'] = 0

                if pod.type in vm.running_pod_list:
                    vm.running_pod_list[pod.type].append(pod)
                else:
                    vm.running_pod_list[pod.type] = []
                    vm.running_pod_list[pod.type].append(pod)
                # for pod in vm.running_pod_list:
                #     # logging.info(
                #     #     "Pod {} is in running list of vm {}".format(pod.id, vm.id))
                pod.start_time = self.clock

                # print(PODS[len(PODS) - 1])
                if pod.type in self.pod_object_dict_by_type:
                    self.pod_object_dict_by_type[pod.type].append(pod)
                else:
                    self.pod_object_dict_by_type[pod.type] = []
                    self.pod_object_dict_by_type[pod.type].append(pod)
                pod.num_current_replicas = len(self.pod_object_dict_by_type[pod.type])

                # if str(pod.type) + "_pod_counter" in pod_counters:
                #     pod_counters[str(pod.type) + "_pod_counter"] += 1
                # else:
                #     pod_counters[str(pod.type) + "_pod_counter"] = 0

                if pod.type not in self.pod_counter:
                    self.pod_counter[pod.type] = 0

                if self.pending_pods[pod.type] > 0:
                    self.pending_pods[pod.type] -= 1
                vm_allocated = True
                break

    def req_scheduler(self, request, rescheduling):
        pod_allocated = False

        rescheduling_try_count = request.reschedule_tries
        # pod_counter_name = request.type + "_pod_counter"

        if rescheduling and rescheduling_try_count >= constants.max_reschedule_tries:
            # logging.info(
            #     "CLOCK: {} worker: {} Request has exceeded the rescheduling count, thus dropping".format(
            #         self.clock, self.worker_id))
            self.dropped_requests.append(request)
            request.start_time = self.clock
            request.finish_time = self.clock
            request.status = "Dropped"
            self.req_wl.write(self.results_sheet_row_counter, 0, request.id)
            self.req_wl.write(self.results_sheet_row_counter, 1, request.type)
            self.req_wl.write(self.results_sheet_row_counter, 2, "None")
            self.req_wl.write(self.results_sheet_row_counter, 3, request.arrival_time)
            self.req_wl.write(self.results_sheet_row_counter, 4, request.start_time)
            self.req_wl.write(self.results_sheet_row_counter, 5, self.clock)
            self.req_wl.write(self.results_sheet_row_counter, 6, "Dropped")

            # with history_lock:
            if request.type in self.sorted_request_history_per_window:
                self.sorted_request_history_per_window[str(request.type)].append(request)
            else:
                self.sorted_request_history_per_window[str(request.type)] = []
                self.sorted_request_history_per_window[str(request.type)].append(request)

            self.results_sheet_row_counter += 1
            self.wb.save("results/" + str(self.worker_id) + "/" + "Results_Episode_" + str(self.episode_no) + ".xls")
        else:
            if request.type in self.pod_object_dict_by_type:
                pod_list_length = len(self.pod_object_dict_by_type[request.type])
                # logging.info("Pods of this type exist, no of pods is : {}".format(pod_list_length))
                # If the global pod counter is greater than the existing pod count, start from the beginning
                if request.type in self.pod_counter:
                    if self.pod_counter[request.type] >= pod_list_length:
                        self.pod_counter[request.type] = 0

                # Allocate requests starting with the pod with highest cpu util
                # sorted_pod_list = sorted(self.pod_object_dict_by_type[request.type], key=self.pod_sorter)
                # for pod in reversed(sorted_pod_list):
                # Allocate requests to pods in a RR manner using the global pod counter for each function type
                for i in range(self.pod_counter[request.type], pod_list_length):
                    pod = self.pod_object_dict_by_type[request.type][i]
                    logging.info("Now considering pod: " + str(pod.id) + "type " + str(pod.type))
                    logging.info("inflight requests: " + str(pod.num_inflight_requests))
                    logging.info(
                        "CLOCK: {} worker: {} pod cpu req : {} pods cpu util: {} pod ram req: {} pod ram util: {}".format(
                            self.clock, self.worker_id, pod.cpu_req,
                            pod.cpu_util,
                            pod.ram_req,
                            pod.ram_util))
                    if ((pod.cpu_req - pod.cpu_util) >= float(self.fn_features[str(request.type) + "_req_MIPS"]/self.fn_features[str(request.type) + "_req_exec_time"])) & ((pod.ram_req - pod.ram_util) >= int(self.fn_features[str(request.type) + "_req_ram"])):
                        request.allocated_pod = pod
                        # logging.info("This pod allocated: pod " + str(pod.id))
                        request.start_time = self.clock
                        request.finish_time = self.clock + float(self.fn_features[str(request.type) + "_req_exec_time"])
                        self.sorted_events.append(
                            defs.EVENT(request.finish_time, constants.finish_request, request))
                        # self.logging.info("Request: {} start time: {} Finish time: {}".format(request.id, request.start_time,
                        #                                                                  request.finish_time))
                        pod_allocated = True
                        pod.num_inflight_requests += 1
                        pod.cpu_util += float(self.fn_features[str(request.type) + "_req_MIPS"]/self.fn_features[str(request.type) + "_req_exec_time"])
                        pod.ram_util += int(self.fn_features[str(request.type) + "_req_ram"])
                        pod.running_req_list.append(request)
                        # with vm_lock:
                        pod.allocated_vm.cpu_used += float(self.fn_features[str(request.type) + "_req_MIPS"]/self.fn_features[str(request.type) + "_req_exec_time"])
                        pod.allocated_vm.ram_used += int(self.fn_features[str(request.type) + "_req_ram"])
                        if request.type in self.pod_counter:
                            self.pod_counter[request.type] += 1
                        else:
                            self.pod_counter[request.type] = 0
                            self.pod_counter[request.type] += 1
                        break

                if not pod_allocated:
                    request.reschedule_tries += 1
                    # logging.info(
                    #     "CLOCK: {} worker: {} Pod is not allocated but rescheduling count not exceeded, thus scheduling again".format(
                    #         self.clock, self.worker_id))
                    self.sorted_events.append(
                        defs.EVENT(self.clock + constants.req_wait_time_to_schedule, constants.re_schedule_request,
                                   request))

            else:
                if request.type in self.pending_pods:
                    request.reschedule_tries += 1
                    # logging.info(
                    #     "CLOCK: {} worker: {} Pod type is not in dict but request type is in pending pods, thus rescheduling request".format(
                    #         self.clock, self.worker_id))
                    self.sorted_events.append(
                        defs.EVENT(self.clock + constants.req_wait_time_to_schedule, constants.re_schedule_request,
                                   request))
                else:
                    self.pending_pods[request.type] = 0
                    self.pending_pods[request.type] += 1
                    request.reschedule_tries += 1
                    # logging.info(
                    #     "CLOCK: {} worker: {} Pod type is not in dict or pending pods thus creating new pod and rescheudling request".format(
                    #         self.clock, self.worker_id))
                    self.sorted_events.append(
                        defs.EVENT(self.clock + constants.pod_creation_time, constants.create_new_pod, request))
                    self.sorted_events.append(
                        defs.EVENT(self.clock + constants.req_wait_time_to_schedule, constants.re_schedule_request,
                                   request))

        self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)
        # Calling pod scaler since after scheduling new requests, pod cpu utilizations would go up and thus they may need to scale up
        self.pod_scaler()

    def req_completion(self, request):

        # logging.info("CLOCK: {} worker: {} Req completion > Request ID: {}".format(self.clock, self.worker_id, request.id))

        request.finish_time = self.clock
        request.allocated_pod.running_req_list.remove(request)
        request.allocated_pod.completed_req_list.append(request)
        # logging.info("CLOCK: {} worker: {} num_inflight_requests before completion: {}".format(self.clock, self.worker_id,
        #                                                                             request.allocated_pod.num_inflight_requests))
        request.allocated_pod.num_inflight_requests -= 1
        # logging.info("CLOCK: {}  worker: {} num_inflight_requests after completion: {}".format(self.clock, self.worker_id,
        #                                                                            request.allocated_pod.num_inflight_requests))
        request.allocated_pod.cpu_util -= float(self.fn_features[str(request.type) + "_req_MIPS"]/self.fn_features[str(request.type) + "_req_exec_time"])
        request.allocated_pod.ram_util -= int(self.fn_features[str(request.type) + "_req_ram"])
        request.allocated_pod.allocated_vm.cpu_used -= float(self.fn_features[str(request.type) + "_req_MIPS"]/self.fn_features[str(request.type) + "_req_exec_time"])
        request.allocated_pod.allocated_vm.ram_used -= int(self.fn_features[str(request.type) + "_req_ram"])
        request.status = "Ok"

        self.req_wl.write(self.results_sheet_row_counter, 0, request.id)
        self.req_wl.write(self.results_sheet_row_counter, 1, request.type)
        self.req_wl.write(self.results_sheet_row_counter, 2, str(request.allocated_pod.id))
        self.req_wl.write(self.results_sheet_row_counter, 3, request.arrival_time)
        self.req_wl.write(self.results_sheet_row_counter, 4, request.start_time)
        self.req_wl.write(self.results_sheet_row_counter, 5, request.finish_time)
        self.req_wl.write(self.results_sheet_row_counter, 6, "Ok")

        if request.type in self.sorted_request_history_per_window:
            self.sorted_request_history_per_window[str(request.type)].append(request)
        else:
            self.sorted_request_history_per_window[str(request.type)] = []
            self.sorted_request_history_per_window[str(request.type)].append(request)

        self.results_sheet_row_counter += 1

        self.wb.save("results/" + str(self.worker_id) + "/" + "Results_Episode_" + str(self.episode_no) + ".xls")

        self.pod_scaler()

        if not request.done:
            request.done = True
            vm = defs.VM(0, len(self.vms), vm_features['vm0_cpu_total_speed'], vm_features['vm0_mem'],
                         vm_features['vm0_cpu_used'], vm_features['vm0_mem_used'],
                         vm_features['vm0_cpu_allocated'],
                         vm_features['vm0_mem_allocated'], vm_features['vm0_price'],
                         vm_features['vm0_nw_bandwidth'],
                         vm_features['vm0_diskio_bandwidth'], [], [])
            pod = defs.POD(self.pod_id, vm, self.fn_features["fn0" + "_name"], 0, 0,
                           self.fn_features["fn0" + "_pod_cpu_req"],
                           self.fn_features["fn0" + "_req_MIPS"],
                           self.fn_features["fn0" + "_pod_ram_req"],
                           self.fn_features["fn0" + "_pod_cpu_util"],
                           self.fn_features["fn0" + "_pod_ram_util"],
                           self.fn_features["fn0" + "_req_per_sec"],
                           self.fn_features["fn0" + "_num_current_replicas"],
                           self.fn_features["fn0" + "_num_max_replicas"],
                           self.fn_features["fn0" + "_inflight_requests"], [], [])
            # Create a request object
            req_obj = defs.REQUEST(request.id, request.chain, request.fn2, pod, self.clock, 0, 0, request.arrival_rate,
                                   0, "initial",
                                   None, True)

            self.sorted_events.append(defs.EVENT(self.clock, constants.schedule_request, req_obj))

            self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)

    def gen_serv_env_state(self, fn_type):
        env_state = np.zeros(148)

        i = 0
        x = 0
        while i < len(self.vms):
            env_state[x] = (self.vms[i].cpu_used / self.vms[i].cpu)
            env_state[x + 1] = (self.vms[i].ram_used / self.vms[i].ram)
            env_state[x + 2] = (self.vms[i].cpu_allocated / self.vms[i].cpu)
            env_state[x + 3] = self.vms[i].cpu / constants.max_vm_cpu
            env_state[x + 4] = self.vms[i].mem_allocated / self.vms[i].ram
            env_state[x + 5] = self.vms[i].ram / constants.max_vm_mem

            replicas = 0
            # print(vms[i].running_pod_list)
            # print(fn_type)

            if fn_type in self.vms[i].running_pod_list:
                replicas = len(self.vms[i].running_pod_list[fn_type])

            # num of replicas of the fn in the vm
            env_state[x + 6] = replicas / constants.max_num_replicas
            x = x + 7
            i += 1

        # the MIPS of a request for the fn
        env_state[x] = self.fn_features[str(fn_type) + "_req_MIPS"] / constants.max_pod_cpu_req
        # the current requested cpu of a pod
        env_state[x + 1] = self.fn_features[str(fn_type) + "_pod_cpu_req"] / constants.max_pod_cpu_req
        # the current requested mem of a pod
        env_state[x + 2] = self.fn_features[str(fn_type) + "_pod_ram_req"] / constants.max_total_pod_mem
        # request concurrency of a pod (affects the average pod resource % used)
        if fn_type in self.pod_object_dict_by_type[fn_type]:
            if len(self.pod_object_dict_by_type[fn_type]) != 0:
                env_state[x + 3] = round(
                    (self.fn_request_rate[fn_type] / len(
                        self.pod_object_dict_by_type[fn_type])) / constants.max_request_rate, 2)

        # logging.info("For state of worker: {} Concurrency of fn {}: {}".format(self.worker_id, fn_type, env_state[x + 2]))

        exec_time = 0
        req_count = 0
        fn_latency = 0
        failed_count = 0
        total_req_count = 0
        failure_rate = 0

        if fn_type in self.sorted_request_history_per_window:
            for req in reversed(self.sorted_request_history_per_window[fn_type]):
                if req.finish_time > self.clock - constants.state_latency_window:
                    total_req_count += 1
                    # self.logging.info(
                    #     "Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time, req.finish_time))
                    if req.status != "Dropped":
                        # self.logging.info("Req id: {}, arrival time: {}, finish time: {}".format(req.id, req.arrival_time,
                        #                                                                     req.finish_time))
                        exec_time += req.finish_time - req.arrival_time
                        req_count += 1
                    else:
                        failed_count += 1

                else:
                    break

        if req_count != 0:
            fn_latency = (exec_time / req_count) / float(self.fn_features[str(fn_type) + "_req_exec_time"])

        if total_req_count != 0:
            failure_rate = failed_count / total_req_count

        # logging.info("latency of fn {}: {}".format(fn_type, fn_latency))

        # average request latency of the fn within the considered window in the past
        env_state[x + 4] = round(fn_latency / constants.max_step_latency_perfn, 2)
        env_state[x + 5] = round(failure_rate, 2)

        total_cpu_util = 0
        total_mem_util = 0
        num_pods = 0

        if fn_type in self.pod_object_dict_by_type:
            num_pods = len(self.pod_object_dict_by_type[fn_type])

        if num_pods != 0:
            for pod in self.pod_object_dict_by_type[fn_type]:
                total_cpu_util += pod.cpu_util / pod.cpu_req
                total_mem_util += pod.ram_util / pod.ram_req

            # average cpu and mem util of fn's pods
            env_state[x + 6] = round(total_cpu_util / num_pods, 2)
            env_state[x + 7] = round(total_mem_util / num_pods, 2)

            # logging.info("Avg pod cpu util of fn {}: {}".format(fn_type, env_state[x + 5]))

        return env_state

    def init_workload(self):
        print("Worker: {} New workload created for new episode {}".format(self.worker_id,
                                                                          self.episode_no % constants.num_wls))
        wbr = xlrd.open_workbook("wl/" + str(self.worker_id) + "/" + "wl" + str(
            self.episode_no % constants.num_wls) + ".xls")
        sheet = wbr.sheet_by_index(0)
        for i in range(sheet.nrows - 1):
            # for i in range(500):
            fn_name = int(sheet.cell_value(i + 1, 0))
            arr_time = sheet.cell_value(i + 1, 1)
            arr_rate = sheet.cell_value(i + 1, 2)
            idr = sheet.cell_value(i + 1, 3)
            status = True

            if not (fn_name in self.fn_types):
                self.fn_types.append(fn_name)
            if not (fn_name in self.fn_request_rate):
                self.fn_request_rate[fn_name] = arr_rate

            running_pod_list = {}
            term_pod_list = []
            run_req_list = []
            term_req_list = []

            # creating a bogus VM and pod object so that a request object can be initialized
            vm = defs.VM(0, len(self.vms), vm_features['vm0_cpu_total_speed'], vm_features['vm0_mem'],
                         vm_features['vm0_cpu_used'], vm_features['vm0_mem_used'],
                         vm_features['vm0_cpu_allocated'],
                         vm_features['vm0_mem_allocated'], vm_features['vm0_price'],
                         vm_features['vm0_nw_bandwidth'],
                         vm_features['vm0_diskio_bandwidth'], running_pod_list, term_pod_list)
            pod = defs.POD(self.pod_id, vm, self.fn_features["fn0" + "_name"], 0, 0,
                           self.fn_features["fn0" + "_pod_cpu_req"],
                           self.fn_features["fn0" + "_req_MIPS"],
                           self.fn_features["fn0" + "_pod_ram_req"],
                           self.fn_features["fn0" + "_pod_cpu_util"],
                           self.fn_features["fn0" + "_pod_ram_util"],
                           self.fn_features["fn0" + "_req_per_sec"],
                           self.fn_features["fn0" + "_num_current_replicas"],
                           self.fn_features["fn0" + "_num_max_replicas"],
                           self.fn_features["fn0" + "_inflight_requests"], run_req_list, term_req_list)
            # Create a request object
            req_obj = defs.REQUEST(idr, 'single', "fn" + str(fn_name), pod, arr_time, 0, 0, arr_rate, 0, "initial",
                                   None, status)

            # Event list (unsorted list) consists of the event received time, event name and the object associated with the event
            self.sorted_events.append(defs.EVENT(arr_time, constants.schedule_request, req_obj))

        self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)

    def init_workload_test(self):
        print("Worker: {} New workload created for new episode {}".format(self.worker_id,
                                                                          self.episode_no % constants.num_wls))
        wbr = xlrd.open_workbook("wl/" + str(self.worker_id) + "/" + "wl" + str(
            self.episode_no % constants.num_wls) + ".xls")
        sheet = wbr.sheet_by_index(0)
        for i in range(sheet.nrows - 1):
            fn_type = str(sheet.cell_value(i + 1, 0))
            # for i in range(500):
            if fn_type == 'single':
                fn1_name = int(sheet.cell_value(i + 1, 1))
                fn2_name = None
                status = True
            else:
                fn_list = sheet.cell_value(i + 1, 1)
                fn_list = re.split("; |, |\]|\[|,", fn_list)
                fn1_name, fn2_name = fn_list[1], fn_list[2]
                status = False
            arr_time = sheet.cell_value(i + 1, 2)
            arr_rate = sheet.cell_value(i + 1, 3)
            idr = sheet.cell_value(i + 1, 4)

            if not (fn1_name in self.fn_types):
                self.fn_types.append(fn1_name)
            # if not (fn_name in self.fn_request_rate):
            #     self.fn_request_rate[fn_name] = arr_rate

            running_pod_list = {}
            term_pod_list = []
            run_req_list = []
            term_req_list = []

            # creating a bogus VM and pod object so that a request object can be initialized
            vm = defs.VM(0, len(self.vms), vm_features['vm0_cpu_total_speed'], vm_features['vm0_mem'],
                         vm_features['vm0_cpu_used'], vm_features['vm0_mem_used'],
                         vm_features['vm0_cpu_allocated'],
                         vm_features['vm0_mem_allocated'], vm_features['vm0_price'],
                         vm_features['vm0_nw_bandwidth'],
                         vm_features['vm0_diskio_bandwidth'], running_pod_list, term_pod_list)
            pod = defs.POD(self.pod_id, vm, self.fn_features["fn0" + "_name"], 0, 0,
                           self.fn_features["fn0" + "_pod_cpu_req"],
                           self.fn_features["fn0" + "_req_MIPS"],
                           self.fn_features["fn0" + "_pod_ram_req"],
                           self.fn_features["fn0" + "_pod_cpu_util"],
                           self.fn_features["fn0" + "_pod_ram_util"],
                           self.fn_features["fn0" + "_req_per_sec"],
                           self.fn_features["fn0" + "_num_current_replicas"],
                           self.fn_features["fn0" + "_num_max_replicas"],
                           self.fn_features["fn0" + "_inflight_requests"], run_req_list, term_req_list)
            # Create a request object
            req_obj = defs.REQUEST(idr, fn_type, "fn" + str(fn1_name), pod, arr_time, 0, 0, arr_rate, 0, "initial",
                                   "fn" + str(fn2_name), status)

            # Event list (unsorted list) consists of the event received time, event name and the object associated with the event
            self.sorted_events.append(defs.EVENT(arr_time, constants.schedule_request, req_obj))

        self.sorted_events = sorted(self.sorted_events, key=self.sorter_events)

    print("Request event creation done")

