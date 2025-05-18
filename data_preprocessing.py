import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
from collections import defaultdict

# 数据路径
TRAIN_PATH = 'train/preliminary'
ITEM_SHARE_FILE = os.path.join(TRAIN_PATH, 'item_share_train_info.json')
USER_INFO_FILE = os.path.join(TRAIN_PATH, 'user_info.json')
ITEM_INFO_FILE = os.path.join(TRAIN_PATH, 'item_info.json')

def load_json_data(file_path):
    """
    加载JSON数据文件并处理可能出现的错误
    """
    try:
        # 逐块读取大型JSON文件
        chunk_size = 10 * 1024 * 1024  # 10MB块大小
        data = []
        is_array_format = False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)  # 重置文件指针
            
            # 检查JSON文件格式是否为数组
            is_array_format = first_char == '['
            
            if is_array_format:
                print(f"正在处理数组格式的JSON文件: {file_path}")
                # 数组格式的JSON文件可以直接加载
                return json.load(f)
            else:
                print(f"正在处理行分隔的JSON文件: {file_path}")
                # 逐行读取并解析
                lines = []
                for line in f:
                    lines.append(line.strip())
                
                # 尝试将行合并为有效的JSON数组
                jsonl_data = "[" + ",".join(lines) + "]"
                try:
                    return json.loads(jsonl_data)
                except json.JSONDecodeError as e:
                    print(f"初始解析失败，尝试修复并重新解析: {str(e)}")
                    # 尝试修复常见JSON错误
                    fixed_data = fix_and_parse_json(lines)
                    if fixed_data:
                        return fixed_data
                    raise Exception(f"无法解析JSON数据: {str(e)}")
    except (IOError, OSError) as e:
        print(f"文件读取错误: {str(e)}")
        raise
    except Exception as e:
        print(f"加载JSON文件时出错: {str(e)}")
        raise

def fix_and_parse_json(lines):
    """
    尝试修复并解析JSON数据
    """
    # 检查是否每行一个JSON对象
    if all(line.startswith('{') and line.endswith('}') for line in lines if line.strip()):
        # 确保每行之间有逗号分隔
        json_text = "[" + ",".join(lines) + "]"
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"修复后仍然解析失败: {str(e)}")
    
    # 尝试修复常见的JSON错误（例如，缺少逗号，错误的引号等）
    try:
        # 将所有行合并，然后尝试修复和解析
        text = "".join(lines)
        
        # 在这里添加修复逻辑，例如:
        # 1. 确保有正确的开始和结束括号
        if not text.startswith('['):
            text = '[' + text
        if not text.endswith(']'):
            text = text + ']'
            
        # 2. 处理缺少的引号和冒号
        # 这里可以添加更复杂的正则表达式匹配来修复特定的模式
        
        try:
            return json.loads(text)
        except:
            # 如果还是失败，尝试更激进的修复方法
            print("尝试更激进的JSON修复方法...")
            # 可以在这里添加更多的修复逻辑
            
    except Exception as e:
        print(f"修复JSON失败: {str(e)}")
    
    return None

def fix_json_file(input_file, output_file=None):
    """
    修复JSON文件中的格式问题
    
    参数:
    input_file: 输入的JSON文件路径
    output_file: 输出的修复后的JSON文件路径。如果为None，则使用原文件名加上"_fixed"后缀
    
    返回:
    修复后的JSON文件路径
    """
    if output_file is None:
        file_name, file_ext = os.path.splitext(input_file)
        output_file = f"{file_name}_fixed{file_ext}"
    
    try:
        print(f"正在修复JSON文件: {input_file}")
        
        # 检查文件是否为JSONL格式（每行一个JSON对象）
        with open(input_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            f.seek(0)  # 重置文件指针
            
            # 检测是否为JSONL格式
            is_jsonl = first_line.startswith('{') and first_line.endswith('}')
            
            if is_jsonl:
                print("检测到JSONL格式，转换为JSON数组...")
                json_lines = []
                for line in f:
                    if line.strip():  # 忽略空行
                        json_lines.append(line.strip())
                
                # 确保所有行都是有效的JSON对象
                fixed_lines = []
                for i, line in enumerate(json_lines):
                    # 移除行末可能的逗号
                    if line.endswith(','):
                        line = line[:-1]
                    
                    # 确保每行都是有效的JSON对象
                    if not (line.startswith('{') and line.endswith('}')):
                        print(f"修复第{i+1}行格式")
                        line = '{' + line + '}'
                    
                    fixed_lines.append(line)
                
                # 创建一个有效的JSON数组
                json_data = "[" + ",".join(fixed_lines) + "]"
            else:
                # 文件可能已经是JSON数组或其他格式，读取全部内容
                json_data = f.read()
                
                # 检查是否已经是数组格式
                if not (json_data.strip().startswith('[') and json_data.strip().endswith(']')):
                    print("将文件转换为JSON数组格式...")
                    json_data = "[" + json_data + "]"
        
        # 使用正则表达式修复常见的JSON错误
        # 1. 修复缺少的逗号
        json_data = re.sub(r'}\s*{', '},{', json_data)
        
        # 2. 修复缺少的冒号
        json_data = re.sub(r'"([^"]+)"\s+([^,\s}]+)', r'"\1": \2', json_data)
        
        # 3. 确保最外层是数组
        if not (json_data.strip().startswith('[') and json_data.strip().endswith(']')):
            json_data = "[" + json_data + "]"
        
        # 验证修复后的JSON是否有效
        try:
            json.loads(json_data)
            print("JSON格式验证成功")
            
            # 写入修复后的文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_data)
                
            print(f"修复后的JSON已保存到: {output_file}")
            return output_file
        except json.JSONDecodeError as e:
            print(f"修复后的JSON仍然无效: {str(e)}")
            
            # 可以在这里添加更多的修复尝试
            # 例如，找到错误位置并尝试手动修复
            error_pos = e.pos
            print(f"错误位置: {error_pos}")
            print(f"错误上下文: {json_data[max(0, error_pos-100):error_pos+100]}")
            
            # 尝试再次修复并保存部分内容
            print("尝试保存可以解析的部分内容...")
            try:
                # 查找最后一个完整的对象
                last_valid_pos = json_data.rfind('}', 0, error_pos)
                if last_valid_pos > 0:
                    partial_data = json_data[:last_valid_pos+1] + "]"
                    json.loads(partial_data)  # 验证部分内容是否有效
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(partial_data)
                    
                    print(f"部分有效的JSON已保存到: {output_file}")
                    return output_file
            except:
                print("无法保存部分内容")
            
            return None
    except Exception as e:
        print(f"修复JSON文件时出错: {str(e)}")
        return None

def convert_to_dataframe(json_data):
    """
    将JSON数据转换为DataFrame
    """
    return pd.DataFrame(json_data)

def split_data(df, test_size=0.2, random_state=42):
    """
    将数据分割为训练集和测试集
    """
    # 随机打乱数据
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # 计算分割点
    split_idx = int(len(df) * (1 - test_size))
    
    # 分割数据
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    return train_df, test_df

def process_batch(file_path, batch_size=10000):
    """
    分批处理大型JSON文件
    
    参数:
    file_path: JSON文件路径
    batch_size: 每批处理的记录数
    
    返回:
    完整的DataFrame
    """
    print(f"开始分批处理文件: {file_path}")
    
    # 初始化结果DataFrame
    result_df = pd.DataFrame()
    
    try:
        # 尝试使用pandas直接读取
        print("尝试使用pandas直接读取JSON文件...")
        try:
            result_df = pd.read_json(file_path)
            print(f"成功读取文件，共{len(result_df)}条记录")
            return result_df
        except Exception as e:
            print(f"pandas直接读取失败: {str(e)}")
            print("尝试使用批处理方式读取...")
        
        # 打开文件进行逐行处理
        with open(file_path, 'r', encoding='utf-8') as f:
            # 检查文件是否为JSON数组格式
            first_char = f.read(1)
            f.seek(0)  # 重置文件指针
            
            if first_char == '[':
                # 对于数组格式，需要特殊处理
                print("检测到JSON数组格式，使用json模块分批加载...")
                all_data = json.load(f)
                
                # 分批处理数据
                batch_count = 0
                for i in range(0, len(all_data), batch_size):
                    batch_data = all_data[i:i+batch_size]
                    batch_df = pd.DataFrame(batch_data)
                    
                    if result_df.empty:
                        result_df = batch_df
                    else:
                        result_df = pd.concat([result_df, batch_df], ignore_index=True)
                    
                    batch_count += 1
                    print(f"已处理第{batch_count}批，当前总记录数: {len(result_df)}")
            else:
                # 对于JSONL格式，逐行处理
                print("检测到JSONL格式，逐行处理...")
                batch_data = []
                batch_count = 0
                
                for line in f:
                    if line.strip():  # 忽略空行
                        try:
                            record = json.loads(line.strip())
                            batch_data.append(record)
                            
                            if len(batch_data) >= batch_size:
                                batch_df = pd.DataFrame(batch_data)
                                
                                if result_df.empty:
                                    result_df = batch_df
                                else:
                                    result_df = pd.concat([result_df, batch_df], ignore_index=True)
                                
                                batch_data = []
                                batch_count += 1
                                print(f"已处理第{batch_count}批，当前总记录数: {len(result_df)}")
                        except json.JSONDecodeError as e:
                            print(f"跳过无效的JSON行: {line[:100]}... 错误: {str(e)}")
                
                # 处理最后一批数据
                if batch_data:
                    batch_df = pd.DataFrame(batch_data)
                    
                    if result_df.empty:
                        result_df = batch_df
                    else:
                        result_df = pd.concat([result_df, batch_df], ignore_index=True)
                    
                    print(f"已处理最后一批，总记录数: {len(result_df)}")
        
        print(f"文件处理完成，总共{len(result_df)}条记录")
        return result_df
    
    except Exception as e:
        print(f"分批处理文件时出错: {str(e)}")
        # 如果处理失败，但已经有部分数据，则返回部分数据
        if not result_df.empty:
            print(f"返回已处理的{len(result_df)}条记录")
            return result_df
        raise

def preprocess_data(item_share_df=None, user_info_df=None, item_info_df=None):
    """
    预处理数据
    
    参数:
        item_share_df: 预先加载的商品分享数据DataFrame，如果为None则从文件加载
        user_info_df: 预先加载的用户信息DataFrame，如果为None则从文件加载
        item_info_df: 预先加载的商品信息DataFrame，如果为None则从文件加载
    
    返回:
        处理后的DataFrame元组 (item_share_df, user_info_df, item_info_df)
    """
    try:
        print("开始数据预处理...")
        
        # 如果未提供预先加载的数据，则从文件加载
        if item_share_df is None:
            # 加载数据
            print(f"加载商品分享训练数据: {ITEM_SHARE_FILE}")
            try:
                item_share_data = load_json_data(ITEM_SHARE_FILE)
                item_share_df = convert_to_dataframe(item_share_data)
            except Exception as e:
                print(f"加载标准方式失败，尝试修复JSON文件: {str(e)}")
                fixed_file = fix_json_file(ITEM_SHARE_FILE)
                if fixed_file:
                    print(f"尝试从修复后的文件加载数据: {fixed_file}")
                    try:
                        item_share_data = load_json_data(fixed_file)
                        item_share_df = convert_to_dataframe(item_share_data)
                    except:
                        print("从修复后的文件加载仍然失败，尝试使用pandas直接读取")
                        try:
                            item_share_df = pd.read_json(fixed_file)
                        except:
                            print("pandas读取失败，尝试分批处理")
                            item_share_df = process_batch(fixed_file)
                else:
                    print("修复文件失败，尝试分批处理原始文件")
                    item_share_df = process_batch(ITEM_SHARE_FILE)
            
            print(f"商品分享数据加载完成，形状: {item_share_df.shape}")
        else:
            print("使用预先加载的商品分享数据")
        
        # 转换时间戳列为日期时间类型
        if 'timestamp' in item_share_df.columns:
            item_share_df['timestamp'] = pd.to_datetime(item_share_df['timestamp'])
        
        if user_info_df is None:
            print(f"加载用户信息数据: {USER_INFO_FILE}")
            user_info_data = load_json_data(USER_INFO_FILE)
            user_info_df = convert_to_dataframe(user_info_data)
            print(f"用户信息数据加载完成，形状: {user_info_df.shape}")
        else:
            print("使用预先加载的用户信息数据")
        
        if item_info_df is None:
            print(f"加载商品信息数据: {ITEM_INFO_FILE}")
            item_info_data = load_json_data(ITEM_INFO_FILE)
            item_info_df = convert_to_dataframe(item_info_data)
            print(f"商品信息数据加载完成，形状: {item_info_df.shape}")
        else:
            print("使用预先加载的商品信息数据")
        
        # 处理缺失值
        # 用户信息
        user_info_df = user_info_df.fillna({
            'user_gender': -1,  # 未知性别
            'user_age': -1,     # 未知年龄
            'user_level': 0     # 默认级别
        })
        
        # 商品信息
        item_info_df = item_info_df.fillna({
            'cate_id': -1,
            'cate_level1_id': -1,
            'brand_id': -1,
            'shop_id': -1
        })
        
        # 商品分享数据
        item_share_df = item_share_df.dropna(subset=['inviter_id', 'item_id', 'voter_id'])
        
        # 检查ID列的数据类型，不再强制转换为整数
        print("检查ID列的数据类型...")
        print(f"inviter_id类型: {item_share_df['inviter_id'].dtype}")
        print(f"item_id类型: {item_share_df['item_id'].dtype}")
        if 'voter_id' in item_share_df.columns:
            print(f"voter_id类型: {item_share_df['voter_id'].dtype}")
        print(f"user_id类型: {user_info_df['user_id'].dtype}")
        print(f"item_id类型: {item_info_df['item_id'].dtype}")
        
        # 确保ID列的类型一致性（训练集和测试集可能有不同的类型）
        # 如果训练集中的ID是整数，但测试集中是字符串，则将训练集转换为字符串
        # 这样可以确保在后续处理中ID的类型一致
        item_share_df['inviter_id'] = item_share_df['inviter_id'].astype(str)
        item_share_df['item_id'] = item_share_df['item_id'].astype(str)
        if 'voter_id' in item_share_df.columns:
            item_share_df['voter_id'] = item_share_df['voter_id'].astype(str)
        user_info_df['user_id'] = user_info_df['user_id'].astype(str)
        item_info_df['item_id'] = item_info_df['item_id'].astype(str)
        
        # 检查缺失值
        print("\n数据缺失值统计:")
        print("商品分享数据缺失值:")
        print(item_share_df.isnull().sum())
        print("\n用户信息数据缺失值:")
        print(user_info_df.isnull().sum())
        print("\n商品信息数据缺失值:")
        print(item_info_df.isnull().sum())
        
        # 返回处理后的DataFrame
        return item_share_df, user_info_df, item_info_df
    
    except Exception as e:
        print(f"预处理数据时出错: {str(e)}")
        raise

if __name__ == "__main__":
    # 测试数据预处理
    try:
        item_share_df, user_info_df, item_info_df = preprocess_data()
        
        # 输出数据的基本信息
        print("\n数据基本信息:")
        print(f"商品分享数据: {len(item_share_df)}行, {item_share_df.columns.tolist()}")
        print(f"用户信息数据: {len(user_info_df)}行, {user_info_df.columns.tolist()}")
        print(f"商品信息数据: {len(item_info_df)}行, {item_info_df.columns.tolist()}")
        
        # 保存处理后的数据
        print("\n保存处理后的数据...")
        item_share_df.to_csv('processed_item_share.csv', index=False)
        user_info_df.to_csv('processed_user_info.csv', index=False)
        item_info_df.to_csv('processed_item_info.csv', index=False)
        
        print("数据预处理完成！")
        
    except Exception as e:
        print(f"运行预处理脚本时出错: {str(e)}") 