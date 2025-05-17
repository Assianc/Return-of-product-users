import datetime
import os

def create_test_file():
    # 确保output/test目录存在
    output_dir = os.path.join('output', 'test')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建文件，指定 UTF-8 编码
    file_path = os.path.join(output_dir, 'test_file.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('这是一个测试文件\n')
        f.write('创建时间: ' + str(datetime.datetime.now()))

if __name__ == '__main__':
    create_test_file()
    print(f'文件已创建在: {os.path.abspath(os.path.join("output", "test", "test_file.txt"))}')