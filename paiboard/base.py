import numpy as np
import os
import json

from paiboard.PAIBoxRuntime.PAIBoxRuntime import PAIBoxRuntime
from paiboard.utils.timeMeasure import *

from paicorelib.framelib.frame_defs import FrameHeader as FH
from paiboard.utils.utils_for_frame import frame_np2txt, frame_remove


class PAIBoard(object):
    def __init__(
        self,
        baseDir: str,
        timestep: int,
        layer_num: int = 0,
        output_delay: int = 0,
        batch_size: int = 1,
        backend: str = "PAIBox",
    ):
        """PAIBoard base module. Inclue method of Encode and Decode.

        Args:
            - baseDir     : Output files from Toolchain. (PAIBox/PAIFLOW)
            - timestep    : timestep for your network.
            - layer_num   : network layer in paicore.
            - output_delay: delay for output frame.
            - batch_size  : batch size for inference.
            - backend     : Choose backend from PAIBox/PAIFLOW.
            - source_chip : which chip you send sync frame to.

        """

        self.baseDir = baseDir
        assert timestep * batch_size <= 256  # batch inference limit
        self.timestep = timestep
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.output_delay = output_delay
        self.backend = backend

        self.max_output_frame_num = 0
        if self.backend == "PAIBox":
            all_coreInfoPath = os.path.join(self.baseDir, "core_params.json")
            with open(all_coreInfoPath, "r", encoding="utf8") as fp:
                all_core_params = json.load(fp)
            for chip_addr in all_core_params:
                chip_coord = eval(chip_addr)
                self.source_chip = chip_coord  # first chip_addr is source
                break
            self.initFrames = PAIBoxRuntime.gen_init_frame(all_core_params)
            self.syncFrames = PAIBoxRuntime.gen_sync_frame(
                self.timestep + self.layer_num, self.source_chip
            )

            inputInfoPath = os.path.join(self.baseDir, "input_proj_info.json")
            with open(inputInfoPath, "r", encoding="utf8") as fp:
                input_proj_info = json.load(fp)

            lcn_list = np.array([], dtype=np.int32)
            for proj in input_proj_info.values():
                lcn_list = np.append(lcn_list, int(proj["lcn"]))
            lcn_list = np.append(lcn_list, 4)
            self.split_timestep = self.timestep
            if batch_size == 1:
                while max(lcn_list) * self.split_timestep > 256:
                    self.split_timestep = int(self.split_timestep / 2)

            self.input_frames_info = PAIBoxRuntime.gen_input_frames_info(
                timestep=self.split_timestep * self.batch_size,
                input_proj_info=input_proj_info,
            )

            outputInfoPath = os.path.join(self.baseDir, "output_dest_info.json")
            with open(outputInfoPath, "r", encoding="utf8") as fp:
                self.output_dest_info = json.load(fp)
            self.output_frames_info = PAIBoxRuntime.gen_output_frames_info(
                timestep=self.timestep * self.batch_size,
                delay=self.output_delay,
                output_dest_info=self.output_dest_info,
            )

        elif backend == "PAIFLOW":
            from runtime import loadAuxNet
            from snn_utils import (
                loadInputFormats,
                loadOutputFormats,
                binary_to_uint64,
                GEN_INIT_SYNC,
            )

            self.baseDir = os.path.join(self.baseDir, "output")
            auxNetDir = os.path.join(self.baseDir, "auxNet")
            self.preNet, self.postNet = loadAuxNet(auxNetDir)

            self.frameFormats, self.frameNums, self.inputNames = loadInputFormats(
                self.baseDir
            )
            self.initFrames, self.syncFrames = GEN_INIT_SYNC(
                self.frameFormats, self.frameNums
            )
            vectorized_binary_to_uint64 = np.vectorize(binary_to_uint64)
            self.frameFormats = vectorized_binary_to_uint64(
                self.frameFormats[self.frameNums[0] : -1]
            )

            self.outDict, self.shapeDict, self.scaleDict, self.mapper = (
                loadOutputFormats(self.baseDir)
            )

            self.coreType = "offline"

        configPath = os.path.join(self.baseDir, "config_all.bin")
        self.configFrames = np.fromfile(configPath, dtype="<u8")

        self.dma_inst = None

    def config(self, *args, **kwargs):
        # dma init & send config frame
        raise NotImplementedError

    def inference(self, *args, **kwargs):
        # send work frame and recvive output frame
        raise NotImplementedError

    def genSpikeFrame(self, input_spike, ifi, init=True):
        if init:
            sync_num = self.split_timestep + self.layer_num
        else:
            sync_num = self.split_timestep
        if isinstance(input_spike, list):
            if self.batch_size > 1:
                # todo: batch_size for multiple input port
                for i in range(len(input_spike)):
                    ifi[i] = ifi[i][0 : input_spike[i].size]
                # todo what happen if two(or more) input's batch different?
                self.batch_ts = input_spike[0].shape[0]
                assert self.batch_ts == input_spike[i].shape[0]
                sync_num = self.batch_ts + self.layer_num
            spikeFrame = np.array([], dtype=np.uint64)
            for i in range(len(input_spike)):
                sf = PAIBoxRuntime.encode(input_spike[i], ifi[i])
                spikeFrame = np.concatenate((spikeFrame, sf))
        else:
            if self.batch_size > 1:
                ifi = [ifi[0][0 : input_spike.size]]
                # 需要实时判断当前batch_size的长度，因为最后一个batch_size可能不满足batch_size
                # 但在生成input_frames_info时，需要用最大的batch_size来生成
                self.batch_ts = input_spike.shape[0]
                assert (
                    self.batch_ts <= self.timestep * self.batch_size
                    and self.batch_ts % self.timestep == 0
                )
                sync_num = self.batch_ts + self.layer_num
            spikeFrame = PAIBoxRuntime.encode(input_spike, ifi[0])
        self.syncFrames = PAIBoxRuntime.gen_sync_frame(sync_num, self.source_chip)
        return spikeFrame

    def genOutputSpike(self, outputFrames, ofi, ts):
        if len(ofi) == 1 and self.batch_size > 1:
            ofi = [
                ofi[0]
                .reshape(-1, self.timestep * self.batch_size)[:, 0 : self.batch_ts]
                .flatten()
            ]

            ts = self.batch_ts
        elif len(ofi) > 1 and self.batch_size > 1:
            # todo
            for i in range(len(ofi)):
                ofi[i] = (
                    ofi[i]
                    .reshape(-1, self.timestep * self.batch_size)[:, 0 : self.batch_ts]
                    .flatten()
                )
            ts = self.batch_ts
        outputSpike = PAIBoxRuntime.decode_spike_less1152(
            timestep=ts,
            oframes=outputFrames,
            oframe_infos=ofi,
            flatten=False,
        )
        if len(ofi) == 1:
            outputSpike = outputSpike[0]
        elif len(ofi) > 1:
            outputSpike_dict = {}
            for index, output_proj_name in enumerate(self.output_dest_info):
                # print(f"output_proj_name: {output_proj_name}")
                outputSpike_dict[output_proj_name] = outputSpike[index]
            return outputSpike_dict
        return outputSpike

    def split_inference(self, input_spike, init=True):
        TimeMeasure = False
        if self.backend == "PAIBox":
            t1 = time.time()
            spikeFrame = self.genSpikeFrame(input_spike, self.input_frames_info, init)
            t2 = time.time()
            time_dict["genSpikeFrame "] = (
                time_dict["genSpikeFrame "] + (t2 - t1) * 1000 * 1000
            )
        elif self.backend == "PAIFLOW":
            from runtime import runPreNetWrap
            from snn_utils import Tensor2FrameWrap

            x = runPreNetWrap(self.preNet, self.inputNames, TimeMeasure, *input_spike)
            spikeFrame = Tensor2FrameWrap(
                x, self.frameFormats, self.inputNames, TimeMeasure
            )

        inputFrames = np.concatenate((spikeFrame, self.syncFrames))
        outputFrames = self.inference(self.initFrames, inputFrames, init)
        outputFrames = frame_remove(outputFrames, FH.WORK_TYPE1)
        if outputFrames.shape[0] > self.max_output_frame_num:
            self.max_output_frame_num = outputFrames.shape[0]
        # print(self.max_output_frame_num, end="")
        # frame_np2txt(outputFrames, self.baseDir + "/outputFrames.txt")
        # frame_np2txt(np.sort(outputFrames), self.baseDir + "/outputFrames_sort.txt")
        if self.backend == "PAIBox":
            t3 = time.time()
            outputSpike = self.genOutputSpike(
                outputFrames, self.output_frames_info, self.timestep
            )
            t4 = time.time()
            time_dict["genOutputSpike"] = (
                time_dict["genOutputSpike"] + (t4 - t3) * 1000 * 1000
            )
            # pred = np.argmax(spike_out)
        elif self.backend == "PAIFLOW":
            from snn_utils import Frame2TensorWrap

            outputFrames_list = []
            for i in range(len(outputFrames)):
                out_str = bin(outputFrames[i])
                outputFrames_list.append(out_str[2:])
            outData, outputSpike = Frame2TensorWrap(
                outputFrames_list,
                self.outDict,
                self.shapeDict,
                self.scaleDict,
                self.mapper,
                self.timestep,
                self.coreType,
                TimeMeasure,
            )
        return outputSpike

    def __call__(self, input_spike):
        if self.split_timestep == self.timestep:
            return self.split_inference(input_spike)
        else:
            for i in range(0, self.timestep, self.split_timestep):
                if i == 0:
                    output_spike = self.split_inference(
                        input_spike[i : i + self.split_timestep]
                    )
                else:
                    output_spike = output_spike + self.split_inference(
                        input_spike[i : i + self.split_timestep], init=False
                    )
            return output_spike

    def record_time(self, full_time):
        # TODO : read register to get core_time
        if self.dma_inst is None:
            pass
        else:
            core_time = self.dma_inst.read_reg(self.dma_inst.US_TIME_TICK)
            record_time(core_time, full_time)

    def perf(self, img_num):
        print_time(img_num)
