import ctypes
import os
import subprocess

import wx
import json
import sys

ctypes.windll.shcore.SetProcessDpiAwareness(1)
p = None
cache_simplify_map = {
    'Off': 0,
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Higher': 4,
    'Highest': 6,
    'Gaming': 8
}

cache_simplify_quality_map = {
    'Off': 100,
    'Low': 99,
    'Medium': 95,
    'High': 90,
    'Higher': 85,
    'Highest': 80,
    'Gaming': 75
}
default_arg = {
    'character': 'lambda_00',
    'input': 2,
    'output': 2,
    'ifm': None,
    'osf': '127.0.0.1:11573',
    'is_extend_movement': False,
    'is_alpha_split': False,
    'is_bongo': False,
    'is_alpha_clean': False,
    'is_eyebrow': False,
    'cache_simplify': 'High',
    'cache_compression': 'High',
    'ram_cache_size': '2gb',
    'vram_cache_size': '2gb',
    'model_select': 'seperable_half',
    'interpolation': 'x2_half',
    'frame_rate_limit': '30',
    'sr': 'Off',
    'device_id': '0',
    'use_tensorrt': False,
    'preset': 'Medium'
}

try:
    f = open('launcher.json')
    args = json.load(f)
    default_arg.update(args)
    f.close()
except:
    pass
finally:
    args = default_arg

p = None
dirPath = 'data/images'
characterList = []

hasRTModel = False
try:
    f = open('data/models/tha3/standard/fp16/decomposer.trt')
    f.close()
    hasRTModel = True
except:
    pass


def refreshList():
    global characterList
    characterList = []
    for item in sorted(os.listdir(dirPath), key=lambda x: -os.path.getmtime(os.path.join(dirPath, x))):
        if '.png' == item[-4:]:
            characterList.append(item[:-4])


refreshList()


class OptionPanel(wx.Panel):
    def __init__(self, parent, title='', desc='', choices=None, mapping=None, type=0, default=None, disabled=False):
        wx.Panel.__init__(self, parent)
        self.type = type
        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = choices
        mainSizer = wx.BoxSizer(wx.HORIZONTAL)
        leftSizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(mainSizer)
        titleText = wx.StaticText(self, wx.ID_ANY, title)
        titleFont = titleText.GetFont()
        titleFont.SetWeight(wx.FONTWEIGHT_SEMIBOLD)
        titleText.SetFont(titleFont)
        leftSizer.Add(titleText, 0, wx.ALL, 0)
        descText = wx.StaticText(self, wx.ID_ANY, desc)
        descFont = descText.GetFont()
        descFont.SetWeight(wx.FONTWEIGHT_EXTRALIGHT)
        descText.SetFont(descFont)
        leftSizer.Add(descText, 0, wx.ALL, 0)
        mainSizer.Add(leftSizer, 1, wx.EXPAND | wx.ALL, 0)
        if self.type == 0:
            self.control = wx.Choice(self, wx.ID_ANY, choices=choices)
            self.control.SetMinSize(wx.Size(300, -1))
            try:
                if default is not None:
                    if self.mapping:
                        self.control.SetSelection(self.mapping.index(default))
                    else:
                        self.control.SetSelection(default)
            except:
                pass
        elif self.type == 1:
            self.control = wx.CheckBox(self, wx.ID_ANY)
            try:
                if default is not None:
                    if self.mapping:
                        self.control.SetValue(self.mapping[default])
                    else:
                        self.control.SetValue(default)

            except:
                pass
        elif self.type == 2:
            self.control = wx.TextCtrl(self, wx.ID_ANY)
            self.control.SetMinSize(wx.Size(300, -1))
            try:
                if default is not None:
                    if self.mapping:
                        self.control.SetValue(self.mapping[default])
                    else:
                        self.control.SetValue(default)
            except:
                pass

        mainSizer.Add(self.control, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 20)
        # self.SetBackgroundColour('#000000') 

    def GetValue(self):
        if self.type == 0:
            ret = self.control.GetSelection()
        elif self.type == 1:
            ret = self.control.GetValue()
        elif self.type == 2:
            ret = self.control.GetValue()
        if self.mapping is not None:
            return self.mapping[ret]
        else:
            return ret


class LauncherPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.number_of_buttons = 0
        self.frame = parent
        self.optionDict = {}
        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        controlSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.widgetSizer = wx.BoxSizer(wx.VERTICAL)

        stEasy = wx.StaticText(self, wx.ID_ANY, "Easy")
        f = stEasy.GetFont()
        f.SetWeight(wx.FONTWEIGHT_HEAVY)
        f = f.MakeLarger()
        stEasy.SetFont(f)
        stVtuber = wx.StaticText(self, wx.ID_ANY, "Vtuber")
        f = stVtuber.GetFont()
        f.SetWeight(wx.FONTWEIGHT_LIGHT)
        f = f.MakeLarger()
        stVtuber.SetFont(f)
        controlSizer.Add(stEasy, 0, wx.ALL | wx.CENTER, 0)
        controlSizer.Add(stVtuber, 1, wx.RIGHT | wx.CENTER, 30)

        self.btnLaunch = wx.Button(self, label="Save && Launch")
        self.btnLaunch.Bind(wx.EVT_BUTTON, self.OnLaunch)
        controlSizer.Add(self.btnLaunch, 0, wx.CENTER | wx.ALL, 10)
        # self.btnAdd = wx.Button(self, label="添加") 
        # self.btnAdd.Bind(wx.EVT_BUTTON, self.OnAddWidget) 
        # controlSizer.Add(self.btnAdd, 0, wx.CENTER | wx.ALL, 5) 
        # self.btnRemove = wx.Button(self, label="删除") 
        # self.btnRemove.Bind(wx.EVT_BUTTON, self.OnRemoveWidget) 
        # controlSizer.Add(self.btnRemove, 0, wx.CENTER | wx.ALL, 5) 
        self.mainSizer.Add(self.widgetSizer, 0, wx.CENTER | wx.ALL, 10)
        self.mainSizer.Add(controlSizer, 0, wx.CENTER | wx.EXPAND | wx.LEFT, 10)
        self.mainSizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)
        self.SetSizer(self.mainSizer)
        self.optionSizer = wx.BoxSizer(wx.VERTICAL)
        self.mainSizer.Add(self.optionSizer, 0, wx.EXPAND | wx.CENTER | wx.ALL, 2)

        def addOption(key, **kwargs):
            kwargs['default'] = args[key]
            t = OptionPanel(self, **kwargs)
            self.optionSizer.Add(t, 0, wx.EXPAND | wx.ALL, 5)
            self.optionDict[key] = t
            return t

        addOption('character', title='Character', desc='选择需要使用的角色图片，位于data/images内',
                  choices=characterList)

        addOption('input', title='Input Device', desc='选择希望使用的面捕数据源',
                  choices=['iFacialMocap', 'OpenSeeFace', 'OpenCV(Webcam)', 'Mouse Input', 'Debug Input'],
                  mapping=[0, 4, 1, 3, 2])
        addOption('ifm', title='iFacialMocap IP:Port', desc='输入iFacialMocap连接使用的IP:端口号', type=2)
        addOption('is_eyebrow', title='Eyebrow', desc='使用眉毛输入，对性能有一定影响\n仅支持iFacialMocap', type=1,
                  default=True)
        addOption('osf', title='OpenSeeFace IP:Port', desc='输入OpenSeeFace连接使用的IP:端口号', type=2)

        addOption('output', title='Output', desc='选择输出目标',
                  choices=['Spout2', 'OBS VirtualCam', 'UnityCapture', 'Debug Output'],
                  mapping=[3, 1, 0, 2])

        addOption('device_id', title='GPU Device', desc='选择需要使用的计算设备(默认0)', choices=['0', '1', '2', '3'])
        addOption('use_tensorrt', title='TensorRT',
                  desc='运行过预构建脚本后开启TensorRT可进一步提升性能\n（未构建成功时不可用，仅NVIDIA显卡支持）',
                  type=1)

        addOption('frame_rate_limit', title='FPS Limit', desc='选择帧率限制目标',
                  choices=['10', '15', '20', '30', '60'])
        addOption('preset', title='Performance Preset', desc='性能预设，注意修改后会覆盖后续配置',
                  choices=['Low', 'Medium', 'High', 'Ultra', 'Custom'])
        addOption('model_select', title='Model Select', desc='选择使用的模型\nStandard Full精度较高性能较低',
                  choices=['Seperable Half', 'Seperable Full', 'Standard Half', 'Standard Full', 'THA4 Half', 'THA4 Full'],
                  mapping=['seperable_half', 'seperable_full', 'standard_half', 'standard_full', 'tha4_half', 'tha4_full'])
        addOption('ram_cache_size', title='RAM Cache Size', desc='分配内存缓存大小\n用于存储最终运算结果',
                  choices=['Off', '1GB', '2GB', '4GB', '8GB', '16GB'],
                  mapping=['0b', '1gb', '2gb', '4gb', '8gb', '16gb'])
        addOption('vram_cache_size', title='VRAM Cache Size', desc='分配显存缓存大小\n用于存储中间结果',
                  choices=['Off', '1GB', '2GB', '4GB', '8GB', '16GB'],
                  mapping=['0b', '1gb', '2gb', '4gb', '8gb', '16gb'])
        addOption('cache_simplify', title='Input Simplify',
                  desc='设置输入简化级别\n输入越简化，缓存命中率越高，动作越不平滑',
                  choices=['Off', 'Low', 'Medium', 'High', 'Higher', 'Highest', 'Gaming'])
        addOption('cache_compression', title='JPEG Compression',
                  desc='设置内存缓存压缩等级\n压缩等级越高，缓存命中率越高，输出质量越差',
                  choices=['Off', 'Low', 'Medium', 'High'])

        addOption('sr', title='SuperResolution', desc='选择使用的超分模型\n由于性能原因，real-esrgan会进行裁切',
                  choices=['Off', 'anime4k_x2', 'waifu2x_x2_half', 'real-esrgan_x4_half', 'waifu2x_x2_full',
                           'real-esrgan_x4_full'])
        addOption('interpolation', title='Frame Interpolation', desc='选择使用的补帧模型',
                  choices=['Off', 'x2_half', 'x3_half', 'x4_half', 'x2_full', 'x3_full', 'x4_full'])

        addOption('is_alpha_clean', title='Alpha Preprocessing',
                  desc='预处理Alpha通道\n代替LayerDiffusion生图后PS蒙版清理操作',
                  type=1)
        addOption('is_extend_movement', title='Extend Movement', desc='基于面捕的XY输入移动、缩放输出图像提升可动性',
                  type=1)
        addOption('is_bongo', title='Bongocat Mode', desc='适当旋转输出以适配Bongocat桌宠', type=1)
        addOption('is_alpha_split', title='Alpha Split', desc='分离透明通道到视频流右侧\n你需要自行进行后续处理',
                  type=1)

        def inputChoice(e=None):
            s = self.optionDict['input'].GetValue()
            if s != 0:
                self.optionSizer.Hide(self.optionDict['ifm'])
                self.optionSizer.Hide(self.optionDict['is_eyebrow'])
            else:
                self.optionSizer.Show(self.optionDict['ifm'])
                self.optionSizer.Show(self.optionDict['is_eyebrow'])
            if s != 4:
                self.optionSizer.Hide(self.optionDict['osf'])
            else:
                self.optionSizer.Show(self.optionDict['osf'])

            self.frame.fSizer.Layout()
            self.frame.Fit()

        self.optionDict['input'].Bind(wx.EVT_CHOICE, inputChoice)
        inputChoice()

        def presetChoice(e=None):
            s = self.optionDict['preset'].GetValue()
            presetControls = [
                self.optionDict['model_select'],
                self.optionDict['ram_cache_size'],
                self.optionDict['vram_cache_size'],
                self.optionDict['cache_simplify'],
                self.optionDict['cache_compression'],
            ]
            presets = {
                'Low': [0, 1, 1, 5, 3],
                'Medium': [1, 1, 1, 4, 2],
                'High': [1, 2, 2, 2, 2],
                'Ultra': [3, 3, 3, 1, 1]
            }

            if s == 'Custom':
                for c in presetControls: self.optionSizer.Show(c)
            else:
                for c in presetControls: self.optionSizer.Hide(c)
            if s in presets:
                opt = presets[s]
                for i in range(5): presetControls[i].control.SetSelection(opt[i])

            self.frame.fSizer.Layout()
            self.frame.Fit()

        self.optionDict['preset'].Bind(wx.EVT_CHOICE, presetChoice)
        presetChoice()

        def onActivate(e):
            global characterList
            tName = self.optionDict['character'].GetValue()
            refreshList()
            self.optionDict['character'].control.SetItems(characterList)
            self.optionDict['character'].control.SetSelection(characterList.index(tName))

        if not hasRTModel:
            self.optionDict['use_tensorrt'].control.SetValue(False)
            self.optionDict['use_tensorrt'].control.Enable(False)
            self.optionDict['use_tensorrt'].control.SetToolTip('需要先构建TensorRT模型')

        self.frame.Bind(wx.EVT_ACTIVATE, onActivate)

    def OnLaunch(self, e):
        global p
        args = {}
        for k in self.optionDict.keys():
            args[k] = self.optionDict[k].GetValue()
        f = open('launcher.json', mode='w')
        json.dump(args, f)
        f.close()
        self.btnLaunch.SetLabelText('Working...')

        if p is not None:
            subprocess.run(['taskkill', '/F', '/PID', str(p.pid), '/T'], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            p = None
            self.btnLaunch.SetLabelText("Save & Launch")
        else:
            run_args = [sys.executable, 'main.py']
            if len(args['character']):
                run_args.append('--character')
                run_args.append(args['character'])

            if args['input'] == 0:
                if len(args['ifm']):
                    run_args.append('--ifm')
                    run_args.append(args['ifm'])
            elif args['input'] == 1:
                run_args.append('--input')
                run_args.append('cam')
            elif args['input'] == 2:
                run_args.append('--debug_input')
            elif args['input'] == 3:
                run_args.append('--mouse_input')
                run_args.append('0,0,' + str(wx.GetDisplaySize().width) + ',' + str(wx.GetDisplaySize().height))
            elif args['input'] == 4:
                if len(args['osf']):
                    run_args.append('--osf')
                    run_args.append(args['osf'])

            if args['output'] == 0:
                run_args.append('--output_webcam')
                run_args.append('unitycapture')
            elif args['output'] == 1:
                run_args.append('--output_webcam')
                run_args.append('obs')
            elif args['output'] == 3:
                run_args.append('--output_webcam')
                run_args.append('spout')
            elif args['output'] == 2:
                run_args.append('--debug')
            if args['is_alpha_split']:
                run_args.append('--alpha_split')
            if args['is_extend_movement']:
                run_args.append('--extend_movement')
                run_args.append('1')
            if args['is_bongo']:
                run_args.append('--bongo')
            if args['is_alpha_clean']:
                run_args.append('--alpha_clean')
            if args['is_eyebrow']:
                run_args.append('--eyebrow')
            if args['cache_simplify'] is not None:
                run_args.append('--simplify')
                run_args.append(str(cache_simplify_map[args['cache_simplify']]))
                run_args.append('--cacher_quality')
                run_args.append(str(cache_simplify_quality_map[args['cache_compression']]))
                if args['cache_simplify'] != 'Off':
                    run_args.append('--use_cacher')
            if args['ram_cache_size'] is not None:
                run_args.append('--cache')
                run_args.append(args['ram_cache_size'])
                run_args.append('--gpu_cache')
                run_args.append(args['vram_cache_size'])

            if args['interpolation'] is not None:
                if not 'Off' == args['interpolation']:
                    run_args.append('--use_interpolation')
                if 'half' in args['interpolation']:
                    run_args.append('--interpolation_half')

                if 'x2' in args['interpolation']:
                    run_args.append('--interpolation_scale')
                    run_args.append('2')
                elif 'x3' in args['interpolation']:
                    run_args.append('--interpolation_scale')
                    run_args.append('3')
                elif 'x4' in args['interpolation']:
                    run_args.append('--interpolation_scale')
                    run_args.append('4')

            if args['model_select'] is not None:
                if 'tha4' in args['model_select']:
                    run_args.append('--use_tha4')
                    if 'half' in args['model_select']:
                        run_args.append('--model_half')
                else:
                    if 'seperable' in args['model_select']:
                        run_args.append('--model_seperable')
                    if 'half' in args['model_select']:
                        run_args.append('--model_half')

            if args['frame_rate_limit'] is not None:
                run_args.append('--frame_rate_limit')
                run_args.append(args['frame_rate_limit'])

            if args['sr'] is not None and args['sr'] != 'Off':
                if 'anime4k' in args['sr']:
                    run_args.append('--anime4k')
                else:
                    run_args.append('--use_sr')
                    if 'x4' in args['sr']:
                        run_args.append('--sr_x4')
                    if 'half' in args['sr']:
                        run_args.append('--sr_half')

            if args['device_id'] is not None:
                run_args.append('--device_id')
                run_args.append(args['device_id'])

            if args['use_tensorrt'] is not None and args['use_tensorrt']:
                run_args.append('--use_tensorrt')
                run_args.append('--model_cache')
                run_args.append('--model_vram_cache')

            run_args.append('--output_size')
            run_args.append('512x512')
            print('Launched: ' + ' '.join(run_args))
            p = subprocess.Popen(run_args)
            self.btnLaunch.SetLabelText('Stop')


class MainFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MainFrame, self).__init__(*args, **kw)
        self.InitUi()

        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, e):
        global p
        if p is not None:
            subprocess.run(['taskkill', '/F', '/PID', str(p.pid), '/T'], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        e.Skip()

    def InitUi(self):
        self.SetTitle("EasyVtuber Launcher")
        self.fSizer = wx.BoxSizer(wx.VERTICAL)
        panel = LauncherPanel(self)
        self.fSizer.Add(panel, 1, wx.EXPAND)
        self.SetSizer(self.fSizer)
        self.SetMinSize(wx.Size(600, 0))
        self.Fit()
        self.Centre()


def main():
    app = wx.App()
    sample = MainFrame(None)

    sample.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()
