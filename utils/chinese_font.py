"""中文字体配置模块

解决 Matplotlib 中文显示问题的通用方案。

要点：
- 尽量不要在导入 pyplot 之后再调用 matplotlib.use()
- 在保存图片（Agg 后端）场景下，建议同时设置 rcParams + 显式传入 FontProperties
"""

import os
import warnings

import matplotlib

warnings.filterwarnings('ignore')

def setup_chinese_font():
    """
    配置matplotlib以正确显示中文
    
    优先级:
    1. 尝试使用系统中的中文字体 (Microsoft YaHei, SimHei等)
    2. 如果系统字体不可用，尝试使用系统字体路径
    3. 设置负号正常显示
    """
    # 使用非交互式后端（避免GUI相关问题）
    # 注意：若 pyplot 已被导入，use() 可能不会生效；这里用 try 避免抛错中断。
    try:
        matplotlib.use('Agg')
    except Exception:
        pass
    
    # 清除字体缓存
    try:
        import matplotlib.font_manager as fm
        # 重建字体缓存
        fm._rebuild()
    except:
        pass
    
    # 方法1: 直接设置字体
    chinese_fonts = [
        'Microsoft YaHei',   # 微软雅黑
        'SimHei',            # 黑体
        'SimSun',            # 宋体
        'KaiTi',             # 楷体
        'FangSong',          # 仿宋
        'STHeiti',           # 华文黑体 (Mac)
        'STSong',            # 华文宋体 (Mac)
        'Heiti SC',          # 黑体-简 (Mac)
        'PingFang SC',       # 苹方-简 (Mac)
        'Noto Sans CJK SC',  # Google开源中文字体 (Linux)
        'WenQuanYi Micro Hei', # 文泉驿微米黑 (Linux)
        'DejaVu Sans',       # 备用字体
    ]
    
    matplotlib.rcParams['font.sans-serif'] = chinese_fonts
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    matplotlib.rcParams['font.family'] = 'sans-serif'
    
    # 方法2: 尝试找到可用的中文字体文件
    font_found = False
    try:
        import matplotlib.font_manager as fm
        
        # Windows字体路径
        windows_font_paths = [
            r'C:\Windows\Fonts\msyh.ttc',      # 微软雅黑
            r'C:\Windows\Fonts\msyhbd.ttc',    # 微软雅黑粗体
            r'C:\Windows\Fonts\simhei.ttf',    # 黑体
            r'C:\Windows\Fonts\simsun.ttc',    # 宋体
        ]
        
        # Linux字体路径
        linux_font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        ]
        
        # Mac字体路径
        mac_font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
        ]
        
        all_font_paths = windows_font_paths + linux_font_paths + mac_font_paths
        
        for font_path in all_font_paths:
            if os.path.exists(font_path):
                # 添加字体到matplotlib
                fm.fontManager.addfont(font_path)
                font_prop = fm.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                
                # 将此字体放在列表最前面
                matplotlib.rcParams['font.sans-serif'] = [font_name] + chinese_fonts
                font_found = True
                print(f"[字体配置] 已加载中文字体: {font_name}")
                break
                
    except Exception as e:
        pass
    
    if not font_found:
        print("[字体配置] 使用默认字体配置")
    
    return font_found


def get_chinese_font():
    """
    获取可用的中文字体属性对象
    
    Returns:
        matplotlib.font_manager.FontProperties 或 None
    """
    try:
        import matplotlib.font_manager as fm
        
        # Windows字体优先
        font_paths = [
            r'C:\Windows\Fonts\msyh.ttc',
            r'C:\Windows\Fonts\simhei.ttf',
            r'C:\Windows\Fonts\simsun.ttc',
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    fm.fontManager.addfont(font_path)
                except Exception:
                    pass
                return fm.FontProperties(fname=font_path)
        
        # 尝试从系统字体中查找
        font_names = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
        for name in font_names:
            try:
                font_path = fm.findfont(fm.FontProperties(family=name))
                if font_path and os.path.exists(font_path):
                    return fm.FontProperties(fname=font_path)
            except:
                continue
                
    except Exception as e:
        pass
    
    return None


# 模块加载时自动配置（尽力而为，不阻塞主流程）
try:
    setup_chinese_font()
except Exception:
    pass
