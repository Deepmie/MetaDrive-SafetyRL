import casadi as ca

def print_all_ipopt_options():
    """打印所有IPOPT选项到文件"""
    # 创建一个简单的优化问题
    x = ca.MX.sym('x')
    nlp = {'x': x, 'f': x**2, 'g': x}
    
    # 设置打印选项文档的选项
    opts = {
        'print_time': 0,
        'ipopt.print_options_documentation': 'yes'
    }
    
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    # 这会打印所有选项到控制台
    print("所有IPOPT选项已打印到控制台")


if __name__ == '__main__':
    print_all_ipopt_options()