#!/usr/bin/env python3
"""
Flight Command Panel — Interactive keyboard control for VUAV drone.

Sử dụng curses để hiển thị dashboard ổn định trong terminal.

Phím tắt (State commands):
  T — Takeoff       L — Land         H — Hover
  N — Navigate      R — RTL          E — Emergency
  M — Manual mode   I — Idle         Q — Quit

Phím điều khiển thủ công (Manual control — chỉ hoạt động ở MANUAL mode):
  W/↑  — Tiến (Forward)       S/↓  — Lùi (Backward)
  A/←  — Trái (Strafe Left)   D/→  — Phải (Strafe Right)
  Z    — Xoay trái (Yaw Left) X    — Xoay phải (Yaw Right)
  PgUp — Bay lên (Ascend)     PgDn — Bay xuống (Descend)
  Space — Dừng (Stop all)
  +/-  — Tăng/giảm tốc độ

Mission & Obstacle Avoidance:
  P — Load & Start Pre-planned Mission
  O — Toggle Obstacle Avoidance ON/OFF
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float64
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State as MavrosState
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import curses
import time
import threading


class FlightCommandPanel(Node):
    def __init__(self):
        super().__init__('flight_command_panel')

        # ── State tracking ─────────────────────────────────
        self.flight_state = 'unknown'
        self.altitude = 0.0
        self.px4_connected = False
        self.px4_armed = False
        self.px4_mode = ''
        self.vision_ok = False
        self.safety_ok = False
        self.command_log = []
        self.lock = threading.Lock()

        # ── Manual control state ───────────────────────────
        self.manual_vx = 0.0
        self.manual_vy = 0.0
        self.manual_vz = 0.0
        self.manual_yaw = 0.0
        self.manual_speed = 0.5      # m/s — adjustable with +/-
        self.manual_yaw_rate = 0.3   # rad/s
        self.manual_alt_speed = 0.3  # m/s

        # Speed presets
        self.speed_presets = [0.2, 0.5, 1.0]
        self.speed_preset_names = ['SLOW', 'MEDIUM', 'FAST']
        self.speed_preset_index = 1  # start at MEDIUM

        # ── Obstacle avoidance state ──────────────────────
        self.obstacle_avoidance_enabled = True
        self.nearest_obstacle_dist = float('inf')
        self.obstacle_estop = False

        # ── Mission state ─────────────────────────────────
        self.mission_name = ''
        self.mission_status = ''          # LOADED, STARTED, PAUSED, etc.
        self.mission_wp_current = 0
        self.mission_wp_total = 0
        self.mission_dist_to_wp = 0.0
        self.mission_action = ''

        # ── Publishers ─────────────────────────────────────
        self.command_pub = self.create_publisher(
            String, '/uav/state_machine/command', 10
        )
        self.manual_vel_pub = self.create_publisher(
            TwistStamped, '/uav/manual_velocity', 10
        )
        # Obstacle avoidance toggle
        self.obs_enable_pub = self.create_publisher(
            Bool, '/uav/obstacle_avoidance/enable', 10
        )
        # Mission command
        self.mission_cmd_pub = self.create_publisher(
            String, '/uav/mission_map/command', 10
        )

        # ── Manual velocity publish timer (20 Hz) ──────────
        self.vel_timer = self.create_timer(0.05, self.publish_manual_velocity)

        # ── Subscribers ────────────────────────────────────
        self.create_subscription(
            String, '/uav/state_machine/state', self.state_cb, 10
        )
        self.create_subscription(
            String, '/uav/state_machine/status', self.status_cb, 10
        )
        self.create_subscription(
            Bool, '/uav/state_machine/safety_ok', self.safety_cb, 10
        )

        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            MavrosState, '/mavros/state', self.mavros_cb, mavros_qos
        )

        # Obstacle avoidance monitoring
        self.create_subscription(
            Float64, '/uav/obstacle_avoidance/nearest_obstacle_dist',
            self.obs_dist_cb, 10
        )
        self.create_subscription(
            Bool, '/uav/obstacle_avoidance/emergency_stop',
            self.obs_estop_cb, 10
        )

        # Mission progress
        self.create_subscription(
            String, '/uav/mission_map/progress',
            self.mission_progress_cb, 10
        )
        self.create_subscription(
            String, '/uav/mission_map/status',
            self.mission_status_cb, 10
        )

    def state_cb(self, msg):
        with self.lock:
            self.flight_state = msg.data

    def status_cb(self, msg):
        with self.lock:
            for part in msg.data.split('|'):
                p = part.strip()
                if p.startswith('alt='):
                    try:
                        self.altitude = float(p.split('=')[1].replace('m', ''))
                    except ValueError:
                        pass
                elif p.startswith('vision='):
                    self.vision_ok = 'OK' in p

    def safety_cb(self, msg):
        with self.lock:
            self.safety_ok = msg.data

    def mavros_cb(self, msg):
        with self.lock:
            self.px4_connected = msg.connected
            self.px4_armed = msg.armed
            self.px4_mode = msg.mode

    def obs_dist_cb(self, msg):
        with self.lock:
            self.nearest_obstacle_dist = msg.data

    def obs_estop_cb(self, msg):
        with self.lock:
            self.obstacle_estop = msg.data

    def mission_progress_cb(self, msg):
        with self.lock:
            try:
                data = json.loads(msg.data)
                self.mission_wp_current = data.get('waypoint', 0)
                self.mission_wp_total = data.get('total', 0)
                self.mission_dist_to_wp = data.get('distance_to_wp', 0.0)
                self.mission_action = data.get('action', '')
                self.mission_name = data.get('mission', self.mission_name)
            except (json.JSONDecodeError, KeyError):
                pass

    def mission_status_cb(self, msg):
        with self.lock:
            try:
                data = json.loads(msg.data)
                self.mission_status = data.get('status', '')
                self.mission_name = data.get('mission', self.mission_name)
                self.mission_wp_total = data.get('waypoints_total', self.mission_wp_total)
            except (json.JSONDecodeError, KeyError):
                pass

    def send_command(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.command_pub.publish(msg)
        with self.lock:
            self.command_log.append((time.strftime('%H:%M:%S'), cmd))
            if len(self.command_log) > 6:
                self.command_log.pop(0)

    def publish_manual_velocity(self):
        with self.lock:
            state = self.flight_state
            vx = self.manual_vx
            vy = self.manual_vy
            vz = self.manual_vz
            yaw = self.manual_yaw

        if state != 'manual_control':
            return

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.z = yaw
        self.manual_vel_pub.publish(msg)

    def set_manual_velocity(self, vx, vy, vz, yaw_rate):
        with self.lock:
            self.manual_vx = vx
            self.manual_vy = vy
            self.manual_vz = vz
            self.manual_yaw = yaw_rate

    def stop_manual(self):
        with self.lock:
            self.manual_vx = 0.0
            self.manual_vy = 0.0
            self.manual_vz = 0.0
            self.manual_yaw = 0.0

    def toggle_obstacle_avoidance(self):
        with self.lock:
            self.obstacle_avoidance_enabled = not self.obstacle_avoidance_enabled
            enabled = self.obstacle_avoidance_enabled
        msg = Bool()
        msg.data = enabled
        self.obs_enable_pub.publish(msg)
        label = 'ON' if enabled else 'OFF'
        self.send_command(f'obs_avoidance_{label.lower()}')

    def send_mission_command(self, cmd: str):
        """Send a command to mission_map_node."""
        msg = String()
        msg.data = cmd
        self.mission_cmd_pub.publish(msg)
        self.send_command(f'mission:{cmd}')

KEY_MAP = {
    ord('t'): 'takeoff',
    ord('T'): 'takeoff',
    ord('l'): 'land',
    ord('L'): 'land',
    ord('h'): 'hover',
    ord('H'): 'hover',
    ord('n'): 'navigate',
    ord('N'): 'navigate',
    ord('r'): 'rtl',
    ord('R'): 'rtl',
    ord('e'): 'emergency',
    ord('E'): 'emergency',
    ord('i'): 'idle',
    ord('I'): 'idle',
    ord('m'): 'manual',
    ord('M'): 'manual',
}

STATE_ICON = {
    'idle':           ('IDLE',           '⏸ '),
    'takeoff':        ('TAKEOFF',        '🚀'),
    'hover':          ('HOVER',          '🟢'),
    'navigation':     ('NAVIGATION',     '🧭'),
    'landing':        ('LANDING',        '🔻'),
    'rtl':            ('RTL',            '🏠'),
    'emergency':      ('EMERGENCY',      '🚨'),
    'manual_control': ('MANUAL CONTROL', '🕹 '),
    'unknown':        ('UNKNOWN',        '❓'),
}

CP_TITLE   = 1
CP_OK      = 2
CP_WARN    = 3
CP_ERR     = 4
CP_DIM     = 5
CP_STATE   = 6
CP_KEY     = 7
CP_BAR     = 8
CP_MANUAL  = 9
CP_MISSION = 10


def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(CP_TITLE, curses.COLOR_CYAN,    -1)
    curses.init_pair(CP_OK,    curses.COLOR_GREEN,   -1)
    curses.init_pair(CP_WARN,  curses.COLOR_YELLOW,  -1)
    curses.init_pair(CP_ERR,   curses.COLOR_RED,     -1)
    curses.init_pair(CP_DIM,   curses.COLOR_WHITE,   -1)
    curses.init_pair(CP_STATE, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(CP_KEY,   curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(CP_BAR,   curses.COLOR_CYAN,    -1)
    curses.init_pair(CP_MANUAL, curses.COLOR_BLACK, curses.COLOR_MAGENTA)
    curses.init_pair(CP_MISSION, curses.COLOR_BLACK, curses.COLOR_YELLOW)


def draw(stdscr, node: FlightCommandPanel):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    W = min(w, 66)  # panel width

    with node.lock:
        state = node.flight_state
        alt = node.altitude
        conn = node.px4_connected
        armed = node.px4_armed
        mode = node.px4_mode or 'N/A'
        vision = node.vision_ok
        safe = node.safety_ok
        logs = list(node.command_log)
        m_vx = node.manual_vx
        m_vy = node.manual_vy
        m_vz = node.manual_vz
        m_yaw = node.manual_yaw
        m_speed = node.manual_speed
        preset_name = node.speed_preset_names[node.speed_preset_index]
        # Obstacle avoidance
        obs_enabled = node.obstacle_avoidance_enabled
        obs_dist = node.nearest_obstacle_dist
        obs_estop = node.obstacle_estop
        # Mission
        mission_name = node.mission_name
        mission_status = node.mission_status
        mission_wp_cur = node.mission_wp_current
        mission_wp_tot = node.mission_wp_total
        mission_dist = node.mission_dist_to_wp
        mission_action = node.mission_action

    is_manual = (state == 'manual_control')
    row = 0

    def put(y, x, text, attr=0):
        nonlocal row
        if 0 <= y < h - 1:
            try:
                stdscr.addnstr(y, x, text, W - x, attr)
            except curses.error:
                pass

    border = '═' * W
    put(row, 0, border, curses.color_pair(CP_TITLE) | curses.A_BOLD)
    row += 1
    title = 'VUAV  Flight  Command  Panel'
    pad = (W - len(title)) // 2
    put(row, 0, ' ' * W, curses.color_pair(CP_STATE) | curses.A_BOLD)
    put(row, pad, title, curses.color_pair(CP_STATE) | curses.A_BOLD)
    row += 1
    put(row, 0, border, curses.color_pair(CP_TITLE) | curses.A_BOLD)
    row += 2

    label, icon = STATE_ICON.get(state, ('UNKNOWN', '?'))
    state_color = CP_OK
    if state in ('emergency',):
        state_color = CP_ERR
    elif state in ('landing', 'unknown'):
        state_color = CP_WARN
    elif state in ('idle',):
        state_color = CP_DIM
    elif state == 'manual_control':
        state_color = CP_MANUAL

    put(row, 2, 'Flight State:  ', curses.A_DIM)
    put(row, 17, f' {icon} {label} ', curses.color_pair(state_color) | curses.A_BOLD)
    row += 2
    put(row, 2, '─' * (W - 4), curses.A_DIM)
    row += 1
    bar_max = 20
    bar_fill = min(int(alt * 5), bar_max)
    bar_str = '█' * bar_fill + '░' * (bar_max - bar_fill)
    put(row, 2, 'Altitude:  ', curses.A_DIM)
    put(row, 13, f'{alt:>6.2f}m', curses.color_pair(CP_BAR) | curses.A_BOLD)
    put(row, 22, f' {bar_str}', curses.color_pair(CP_BAR))
    row += 1

    put(row, 2, 'PX4:       ', curses.A_DIM)
    if conn:
        put(row, 13, '● CONNECTED', curses.color_pair(CP_OK))
    else:
        put(row, 13, '● DISCONNECTED', curses.color_pair(CP_ERR))
    row += 1

    put(row, 2, 'Armed:     ', curses.A_DIM)
    if armed:
        put(row, 13, '  ARMED', curses.color_pair(CP_OK) | curses.A_BOLD)
    else:
        put(row, 13, '  DISARMED', curses.color_pair(CP_WARN))
    row += 1

    put(row, 2, 'Mode:      ', curses.A_DIM)
    mode_cp = CP_OK if mode == 'OFFBOARD' else CP_WARN
    put(row, 13, f'  {mode}', curses.color_pair(mode_cp))
    row += 1

    put(row, 2, 'Vision:    ', curses.A_DIM)
    if vision:
        put(row, 13, '● OK', curses.color_pair(CP_OK))
    else:
        put(row, 13, '● FAIL', curses.color_pair(CP_ERR))
    row += 1

    put(row, 2, 'Safety:    ', curses.A_DIM)
    if safe:
        put(row, 13, '● SAFE', curses.color_pair(CP_OK))
    else:
        put(row, 13, '● CHECK', curses.color_pair(CP_WARN))
    row += 1

    put(row, 2, '─' * (W - 4), curses.A_DIM)
    row += 1

    put(row, 2, 'Obstacle Avoidance:', curses.A_BOLD)
    row += 1

    put(row, 4, 'Status: ', curses.A_DIM)
    if obs_enabled:
        put(row, 12, '● ON', curses.color_pair(CP_OK) | curses.A_BOLD)
    else:
        put(row, 12, '● OFF', curses.color_pair(CP_WARN) | curses.A_BOLD)

    if obs_dist < 100:
        dist_str = f'{obs_dist:.1f}m'
        dist_cp = CP_ERR if obs_dist < 1.0 else (CP_WARN if obs_dist < 2.0 else CP_OK)
    else:
        dist_str = 'clear'
        dist_cp = CP_OK
    put(row, 20, f'Nearest: ', curses.A_DIM)
    put(row, 29, dist_str, curses.color_pair(dist_cp) | curses.A_BOLD)

    if obs_estop:
        put(row, 40, 'E-STOP', curses.color_pair(CP_ERR) | curses.A_BOLD)
    row += 1
    put(row, 2, '─' * (W - 4), curses.A_DIM)
    row += 1
    put(row, 2, 'Mission:', curses.A_BOLD)
    row += 1

    if mission_name:
        put(row, 4, f'Name: ', curses.A_DIM)
        put(row, 10, mission_name, curses.color_pair(CP_TITLE) | curses.A_BOLD)
        row += 1

        put(row, 4, 'Status: ', curses.A_DIM)
        status_cp = CP_OK
        if mission_status in ('STARTED', 'RESUMED'):
            status_cp = CP_OK
        elif mission_status == 'PAUSED':
            status_cp = CP_WARN
        elif mission_status in ('COMPLETED',):
            status_cp = CP_TITLE
        elif mission_status in ('ABORTED', 'ERROR'):
            status_cp = CP_ERR
        elif mission_status == 'LOADED':
            status_cp = CP_DIM
        put(row, 12, mission_status or 'N/A', curses.color_pair(status_cp) | curses.A_BOLD)
        row += 1

        if mission_wp_tot > 0:
            progress_pct = int(mission_wp_cur / mission_wp_tot * 100)
            prog_bar_w = 15
            prog_fill = int(progress_pct / 100 * prog_bar_w)
            prog_str = '█' * prog_fill + '░' * (prog_bar_w - prog_fill)
            put(row, 4, f'WP: {mission_wp_cur}/{mission_wp_tot}', curses.A_DIM)
            put(row, 16, f' {prog_str} ', curses.color_pair(CP_BAR))
            put(row, 33, f'{progress_pct}%', curses.color_pair(CP_BAR) | curses.A_BOLD)
            if mission_dist > 0:
                put(row, 39, f'Dist: {mission_dist:.1f}m', curses.A_DIM)
            row += 1
            if mission_action:
                put(row, 4, f'Action: {mission_action}', curses.A_DIM)
                row += 1
    else:
        put(row, 4, '(no mission loaded)', curses.A_DIM)
        row += 1

    put(row, 2, '─' * (W - 4), curses.A_DIM)
    row += 1

    if is_manual:
        put(row, 2, '🕹  MANUAL CONTROL ACTIVE', curses.color_pair(CP_MANUAL) | curses.A_BOLD)
        row += 1

        put(row, 2, f'Speed: {m_speed:.1f} m/s [{preset_name}]', curses.color_pair(CP_BAR))
        put(row, 35, '[+] faster  [-] slower', curses.A_DIM)
        row += 1

        vx_color = CP_OK if abs(m_vx) > 0.01 else CP_DIM
        vy_color = CP_OK if abs(m_vy) > 0.01 else CP_DIM
        vz_color = CP_OK if abs(m_vz) > 0.01 else CP_DIM
        yr_color = CP_OK if abs(m_yaw) > 0.01 else CP_DIM
        put(row, 2, 'Vx:', curses.A_DIM)
        put(row, 6, f'{m_vx:+.2f}', curses.color_pair(vx_color) | curses.A_BOLD)
        put(row, 14, 'Vy:', curses.A_DIM)
        put(row, 18, f'{m_vy:+.2f}', curses.color_pair(vy_color) | curses.A_BOLD)
        put(row, 26, 'Vz:', curses.A_DIM)
        put(row, 30, f'{m_vz:+.2f}', curses.color_pair(vz_color) | curses.A_BOLD)
        put(row, 38, 'Yaw:', curses.A_DIM)
        put(row, 43, f'{m_yaw:+.2f}', curses.color_pair(yr_color) | curses.A_BOLD)
        row += 1

        fwd = '▲' if m_vx > 0.01 else ('▼' if m_vx < -0.01 else '·')
        lat = '◄' if m_vy < -0.01 else ('►' if m_vy > 0.01 else '·')
        alt_ind = '⬆' if m_vz > 0.01 else ('⬇' if m_vz < -0.01 else '·')
        yaw_ind = '↺' if m_yaw < -0.01 else ('↻' if m_yaw > 0.01 else '·')

        put(row, 2, f'  Direction: {fwd} {lat}   Alt: {alt_ind}   Yaw: {yaw_ind}',
            curses.color_pair(CP_BAR))
        row += 1

        put(row, 2, '─' * (W - 4), curses.A_DIM)
        row += 1
        put(row, 2, 'Movement:', curses.A_BOLD)
        row += 1
        put(row, 4, 'W/↑ Forward   S/↓ Backward   Space Stop', curses.A_DIM)
        row += 1
        put(row, 4, 'A/← Left      D/→ Right', curses.A_DIM)
        row += 1
        put(row, 4, 'Z Yaw Left    X Yaw Right', curses.A_DIM)
        row += 1
        put(row, 4, 'PgUp Ascend   PgDn Descend', curses.A_DIM)
        row += 1
    else:
        put(row, 2, 'Control Mode: ', curses.A_DIM)
        put(row, 16, 'AUTONOMOUS', curses.color_pair(CP_OK))
        put(row, 28, '  Press [M] for Manual', curses.A_DIM)
        row += 1

    row += 1
    put(row, 2, '─' * (W - 4), curses.A_DIM)
    row += 1
    put(row, 2, 'Commands:', curses.A_BOLD)
    row += 1

    keys = [
        ('T', 'Takeoff', CP_BAR),
        ('L', 'Land',    CP_WARN),
        ('H', 'Hover',   CP_OK),
        ('N', 'Nav',     CP_TITLE),
        ('M', 'Manual',  CP_MANUAL if not is_manual else CP_OK),
    ]

    col = 3
    for k, lbl, cp in keys:
        put(row, col, f'[', curses.A_DIM)
        put(row, col + 1, k, curses.color_pair(cp) | curses.A_BOLD)
        put(row, col + 2, f'] {lbl}', curses.A_DIM)
        col += len(lbl) + 5
    row += 1

    keys2 = [
        ('R', 'RTL',     CP_DIM),
        ('E', 'Emerg',   CP_ERR),
        ('I', 'Idle',    CP_DIM),
        ('P', 'Mission', CP_MISSION),
        ('O', 'ObsAvd',  CP_OK if obs_enabled else CP_WARN),
        ('Q', 'Quit',    CP_DIM),
    ]

    col = 3
    for k, lbl, cp in keys2:
        put(row, col, f'[', curses.A_DIM)
        put(row, col + 1, k, curses.color_pair(cp) | curses.A_BOLD)
        put(row, col + 2, f'] {lbl}', curses.A_DIM)
        col += len(lbl) + 5
    row += 2
    put(row, 2, '─' * (W - 4), curses.A_DIM)
    row += 1
    put(row, 2, 'Command Log:', curses.A_BOLD)
    row += 1

    if logs:
        for ts, cmd in logs[-5:]:
            _, icon = STATE_ICON.get(cmd, ('', ''))
            put(row, 4, f'{ts}  → {cmd}', curses.A_DIM)
            row += 1
    else:
        put(row, 4, '(no commands sent yet)', curses.A_DIM)
        row += 1

    row += 1
    put(row, 0, border, curses.color_pair(CP_TITLE) | curses.A_BOLD)

    stdscr.refresh()


def curses_main(stdscr, node: FlightCommandPanel):
    init_colors()
    curses.curs_set(0)         # hide cursor
    stdscr.nodelay(True)       # non-blocking getch
    stdscr.timeout(100)        # refresh every 100ms (faster for manual control)
    active_keys = set()

    def get_move_keys(node):
        spd = node.manual_speed
        yaw = node.manual_yaw_rate
        alt = node.manual_alt_speed
        return {
            ord('w'):         ( spd,    0,    0,    0),
            curses.KEY_UP:    ( spd,    0,    0,    0),
            ord('s'):         (-spd,    0,    0,    0),
            curses.KEY_DOWN:  (-spd,    0,    0,    0),
            ord('a'):         (   0, -spd,    0,    0),
            curses.KEY_LEFT:  (   0, -spd,    0,    0),
            ord('d'):         (   0,  spd,    0,    0),
            curses.KEY_RIGHT: (   0,  spd,    0,    0),
            ord('z'):         (   0,    0,    0, -yaw),
            ord('x'):         (   0,    0,    0,  yaw),
            curses.KEY_PPAGE: (   0,    0,  alt,    0),  # Page Up → Ascend
            curses.KEY_NPAGE: (   0,    0, -alt,    0),  # Page Down → Descend
        }

    while True:
        draw(stdscr, node)

        key = stdscr.getch()
        if key == -1:
            with node.lock:
                is_manual = (node.flight_state == 'manual_control')
            if is_manual and active_keys:
                active_keys.clear()
                node.stop_manual()
            continue

        if key in (ord('q'), ord('Q'), 27):  # q or ESC
            break
        if key in (ord('p'), ord('P')):
            with node.lock:
                ms = node.mission_status
            if ms in ('', 'COMPLETED', 'ABORTED', 'ERROR', 'LOADED'):
                node.send_mission_command('start')
            elif ms == 'PAUSED':
                node.send_mission_command('resume')
            elif ms in ('STARTED', 'RESUMED'):
                node.send_mission_command('pause')
            continue

        if key in (ord('o'), ord('O')):
            node.toggle_obstacle_avoidance()
            continue

        if key in KEY_MAP:
            node.send_command(KEY_MAP[key])
            continue

        with node.lock:
            is_manual = (node.flight_state == 'manual_control')

        if not is_manual:
            continue

        if key == ord(' '):
            active_keys.clear()
            node.stop_manual()
            continue

        if key in (ord('+'), ord('=')):
            with node.lock:
                if node.speed_preset_index < len(node.speed_presets) - 1:
                    node.speed_preset_index += 1
                    node.manual_speed = node.speed_presets[node.speed_preset_index]
            continue

        if key in (ord('-'), ord('_')):
            with node.lock:
                if node.speed_preset_index > 0:
                    node.speed_preset_index -= 1
                    node.manual_speed = node.speed_presets[node.speed_preset_index]
            continue

        move_keys = get_move_keys(node)
        if key in move_keys:
            active_keys.add(key)
            vx, vy, vz, yaw = 0.0, 0.0, 0.0, 0.0
            for ak in active_keys:
                if ak in move_keys:
                    dx, dy, dz, dyaw = move_keys[ak]
                    vx += dx
                    vy += dy
                    vz += dz
                    yaw += dyaw
            node.set_manual_velocity(vx, vy, vz, yaw)


def main(args=None):
    rclpy.init(args=args)
    node = FlightCommandPanel()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        curses.wrapper(lambda stdscr: curses_main(stdscr, node))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print('Flight Command Panel closed.')

if __name__ == '__main__':
    main()
