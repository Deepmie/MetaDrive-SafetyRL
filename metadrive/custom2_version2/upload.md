我想让PPO做上层决策，然后MPC做底层控制，即让PPO输出参考轨迹，MPC接管并且作出相应的控制。为此设计了一个控制器类：

```python
class Controller:
    def __init__(self, env: ParallelEnv, config: ControllerConfig):
        self.env = env
        self.config = config
        self.control_values_prev = np.zeros([self.config.n_process, self.config.control_dim])
        self._build_mpc_controller()

    def control(self, actions: ndarray, dones: ndarray) -> Tuple[ndarray, ndarray]: # [env_num, action_dim]
        # 获得初始状态
        vehicle_states_init = self._get_vehicle_state()
        
        # control_values: [[delta, a], ...]
        control_values = np.empty([self.config.n_process, self.config.control_dim])
        state_values = np.empty([self.config.n_process, self.config.vehicle_state_dim])

        for idx, state in enumerate(vehicle_states_init): 
            x0 = np.array([state.x, state.y, state.v, state.theta])
            z_ref = actions[idx, :]
            u_prev = self.control_values_prev[idx, :]
            u, x = self.mpc_controller(x0, z_ref, u_prev)
            u, x = cast(ndarray, u), cast(ndarray, x)
            assert np.linalg.norm(x0 - x[0, :]) < 0.1, f'x0 != x[0, :], x0={x0.tolist()}, x[0, :]={x[0, :].tolist()}'
            control_values[idx, :] = u[0, ::-1]
            state_values[idx, :] = x[1, :] # 第0项是初始状态
        
        self.control_values_prev = (1 - dones.reshape(-1, 1)) * control_values
        return control_values, state_values

    def _build_mpc_controller(self):
        metadata = self.env.get_metadata()
        mpc_config = MPConfig()
        self.mpc_controller = MPC(mpc_config, metadata)

    def _get_vehicle_state(self) -> List[VehicleState]:
        vehicle_states: List[Dict] = self.env.get_state()
        vehicle_states_extracted: List[VehicleState] = list()
        for state in vehicle_states:
            vehicle_states_extracted.append(self._extract_row_state(state))
        return vehicle_states_extracted

    def _extract_row_state(self, state: Dict) -> VehicleState:
        vehicle_state = VehicleState()
        pos = state.get('position', None)
        vel = state.get('velocity')
        
        vehicle_state.x = pos[0]
        vehicle_state.y = pos[1]
        vehicle_state.v = np.linalg.norm(vel, 2) # 取速度的二范数
        vehicle_state.theta = state.get('heading_theta', None)
        vehicle_state.a = state.get('throttle_brake', None)
        vehicle_state.delta = state.get('steering', None)
        return vehicle_state


class EvalController:
    def __init__(self, env: MetaDriveEnv, config: ControllerConfig):
        self.env = env
        self.config = config
        self.control_value_prev = np.zeros([self.config.control_dim])

    def reset(self):
        self._build_mpc_controller()

    def control(self, action: ndarray, done: ndarray) -> Tuple[ndarray, ndarray]: # [env_num, action_dim]
        # 获得初始状态
        vehicle_state_init = self._get_vehicle_state()
        
        # control_values: [[delta, a], ...]
        control_value = np.empty([self.config.control_dim])
        state_value = np.empty([self.config.vehicle_state_dim])

        x0 = np.array([vehicle_state_init.x, vehicle_state_init.y, vehicle_state_init.v, vehicle_state_init.theta])
        z_ref = action
        u_prev = self.control_value_prev
        u, x = self.mpc_controller(x0, z_ref, u_prev)
        u, x = cast(ndarray, u), cast(ndarray, x)
        assert np.sum(x0 - x[0, :]) < 0.1, f'x0 != x[0, :], x0={x0.tolist()}, x[0, :]={x[0, :].tolist()}'
        control_value = u[0, ::-1]
        state_value = x[1, :] # 第0项是初始状态
        
        self.control_value_prev = (1 - done) * control_value
        return control_value, state_value

    def _build_mpc_controller(self):
        metadata = self.get_metadata()
        mpc_config = MPConfig()
        self.mpc_controller = MPC(mpc_config, metadata)

    def _get_vehicle_state(self) -> VehicleState:
        vehicle_state: Dict = self.get_state()
        return self._extract_row_state(vehicle_state)
    
    def get_metadata(self) -> Dict:
        agent: DefaultVehicle = self.env.agent
        return {'l': agent.FRONT_WHEELBASE + agent.REAR_WHEELBASE}
    
    def get_state(self):
        agent: DefaultVehicle = self.env.agent
        return agent.get_state()

    def _extract_row_state(self, state: Dict) -> VehicleState:
        vehicle_state = VehicleState()
        pos = state.get('position', None)
        vel = state.get('velocity')
        
        vehicle_state.x = pos[0]
        vehicle_state.y = pos[1]
        vehicle_state.v = np.linalg.norm(vel, 2) # 取速度的二范数
        vehicle_state.theta = state.get('heading_theta', None)
        vehicle_state.a = state.get('throttle_brake', None)
        vehicle_state.delta = state.get('steering', None)
        return vehicle_state
```

控制调用的MPC类设计如下，其继承OPC类
```python
class OCP(ABC):
    def __init__(self, config: OCPconfig, metadata: Dict):
        self.config = config
        self.metadata = metadata
        self._is_first_created = True
        self._define_state_update_equation()

    def __call__(self, *args, **kwargs) -> ndarray:
        if self._is_first_created:
            opts = {'ipopt.print_level': 0, 'print_time': 0}
            self._caculate_cost_and_conditions()
            self._build_numeric_problem()

            self.solver = ca.nlpsol('solver', 'ipopt', self._nlp, opts)
            self._is_first_created = False
        
        # 添加p的参数
        p_args = []
        
        for a in list(args):
            p_args.extend(self._converto_list(a))
        
        for key in sorted(kwargs.keys()):
            p_args.extend(self._converto_list(kwargs[key]))
        
        p_args = np.array(p_args, dtype=np.float32)
        return self.solver(**self._num_prob, p=p_args)

    @ abstractmethod
    def _build_numeric_problem(self):
        '''
        构建数值问题, 从ca的符号变量转化到数值变量
        '''
    
    @ abstractmethod
    def _caculate_cost_and_conditions(self):
        '''
        计算代价和约束条件的模块
        '''

    @ abstractmethod
    def _define_state_update_equation(self):
        '''
        定义状态转移方程用
        '''

    def _converto_dm(self, args) -> ca.DM:
        if isinstance(args, ndarray) or isinstance(args, list):
            return ca.DM(args)
        elif isinstance(args, float) or isinstance(args, int) or isinstance(args, bool):
            return ca.DM([args])
        return args
    
    def _converto_list(self, args) -> List:
        if isinstance(args, ndarray):
            return args.tolist()
        if isinstance(args, int) or isinstance(args, float) or isinstance(args, bool):
            return [args]
        return args
```

```python
class MPC(OCP):
    def __init__(self, config: MPConfig, metadata: Dict):
        '''
        config: MPC的配置,
        metadata: 一些常量配置, 不应该在update中改变
        '''
        self.l = metadata.get('l', None)
        super(MPC, self).__init__(config, metadata)
    
    def __call__(self, x0, z_ref, u_prev) -> Tuple[ndarray, ndarray]:
        '''
        x0: 当前车辆的状态: [横坐标x, 纵坐标y, 车辆速度v, 车辆角度theta],
        z_ref: 车辆的跟踪轨迹: [参考速度v_ref, 参考角度theta_ref],
        u_prev: 上一次的控制值: [加速度a_prev, 转向角delta_prev]
        '''
        res: Dict = super(MPC, self).__call__(x0, z_ref, u_prev)
        return self._parse_result(res)

    def _build_numeric_problem(self):
        # 设定决策变量的初值, 和值的上下界
        w0 = np.zeros(self._nlp_metadata['w_dim'])
        lbw = np.concatenate([
            np.tile([self.config.a_min, self.config.delta_min], self._nlp_metadata['u_dim'] // 2),
            -np.inf * np.ones(self._nlp_metadata['x_dim']),
        ])
        ubw = np.concatenate([
            np.tile([self.config.a_max, self.config.delta_max], self._nlp_metadata['u_dim'] // 2),
            np.inf * np.ones(self._nlp_metadata['x_dim']),
        ])

        # 设定约束条件的上下界
        lbg = np.zeros(self._nlp_metadata['g_dim'])
        ubg = np.zeros(self._nlp_metadata['g_dim'])

        self._num_prob = {'x0': w0, 'lbx': lbw, 'ubx': ubw, 'lbg': lbg, 'ubg': ubg, }

    def _caculate_cost_and_conditions(self):
        # ------- 添加决策变量 ----- #
        U = ca.MX.sym('U', self.config.nu * self.config.mu)
        X = ca.MX.sym('X', self.config.nx * (self.config.np + 1))

        # -------- 添加常量 ------- #
        x0 = ca.MX.sym('x0', self.config.nx)
        z_ref = ca.MX.sym('z_ref', 2)
        u_prev = ca.MX.sym('u_prev', self.config.nu)

        Q  = DM(np.diag([3, 12])) # error
        R  = DM(np.diag([5, 5]))  # cost
        Rd = DM(np.diag([5, 5]))  # delta

        g = list()
        cost = 0
        
        g.append(X[0: self.config.nx] - x0)
        u_prev_sym = u_prev
        for k in range(self.config.np):
            xk = X[k * self.config.nx: (k+1) * self.config.nx]
            
            if k < self.config.mu:
                uk = U[k * self.config.nu: (k+1) * self.config.nu]
            else:
                uk = U[(self.config.mu - 1) * self.config.nu: self.config.mu * self.config.nu]
            
            zk = xk[2::] # 取x的后两项
            
            # 状态更新
            xk_next = self._f(xk, uk)

            # 约束和代价更新
            g.append(X[(k+1) * self.config.nx: (k+2) * self.config.nx] - xk_next)
            
            zk_e = z_ref - zk
            cost += ca.mtimes([zk_e.T, Q, zk_e])
            cost += ca.mtimes([uk.T, R, uk])
            
            du = uk - u_prev_sym
            cost += ca.mtimes([du.T, Rd, du])
            
            # 控制变量更新
            u_prev_sym = uk
        
        self._nlp = \
        {
            'x': ca.vertcat(U, X),
            'f': cost,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(x0, z_ref, u_prev)
        }

        self._nlp_metadata = \
        {
            'w_dim': U.shape[0] + X.shape[0],
            'u_dim': U.shape[0],
            'x_dim': X.shape[0],
            'g_dim': (self.config.np + 1) * self.config.nx,
        }

    def _define_state_update_equation(self):
        x = ca.MX.sym('x', self.config.nx)
        u = ca.MX.sym('u', self.config.nu)

        # 状态更新方程
        beta = ca.atan(1 / 2 * ca.tan(u[1]))
        x_next = ca.vertcat(
            x[0] + (x[2] * ca.cos(x[3] + beta)) * self.config.Ts,
            x[1] + (x[2] * ca.sin(x[3] + beta)) * self.config.Ts,
            x[2] + u[0] * self.config.Ts,
            x[3] + x[2] / self.l * ca.sin(beta) * self.config.Ts,
        )
        self._f = ca.Function('f', [x, u], [x_next])

    def _parse_result(self, res: Dict) -> Tuple[ndarray, ndarray]:
        '''
        Args:
        res: 求解器solver返回的结果
        Returns:
        元组, 第一项是u, 形状是(np, nu); 第二项是x, 形状是(np+1, nx).
        '''

        w: ndarray = res.get('x', None).full().flatten()
        u: ndarray = w[0: self._nlp_metadata['u_dim']].reshape(self.config.mu, self.config.nu)
        x: ndarray = w[self._nlp_metadata['u_dim']::].reshape(self.config.np + 1, self.config.nx)
        return (u, x)
```

在PPO中的补丁如下：
```python
# 上层决策
actions, log_probs, values = self.policy.select_action(self._last_obss)

# 底层控制
control_values, _ = self.controller.control(actions, self._last_dones)

obs_nexts, rewards, dones, step_infos = self.env.step(control_values)
```

帮我检查控制器或者MPC算法的实现有没有问题。


