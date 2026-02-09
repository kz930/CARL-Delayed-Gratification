import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

import random
from math import exp
import numpy as np
from scipy.stats import norm
import time
import copy

from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions

"""
Kary Zheng
Current version — delayed gratification robot
FEB 9TH
(Testing)
"""

# ------------------ CONSTANTS ------------------

ACTIONS = 2
LEFT = 0
RIGHT = 1

STATES = 4
EXPM_LR = 0
EXPM_RL = 1
CTRL_LR = 2
CTRL_RL = 3

LIVE_RWD = 5.0
DEAD_RWD = 1.0
UNOBTAINABLE_RWD = 0.0

ALPHA = 0.10
BETA = 1.0
GAMMA = 0.99

# ------------------ MAP COORDINATES ------------------

HOME_X = -0.023635001853108406
HOME_Y = -0.023635001853108406

LEFT_CHAMBER  = (1.9318245649,  0.5751032829)
RIGHT_CHAMBER = (1.9457954168, -0.2844115793)

# ------------------ Q TABLE ------------------

qTbl = [[0.0 for _ in range(ACTIONS)] for _ in range(STATES)]

# ------------------ RL FUNCTIONS ------------------

def prob_wait(tim):
    mean = 70
    std_dev = 20
    beta_weight = 2
    return 1 / np.exp(beta_weight * norm.cdf(tim, loc=mean, scale=std_dev))


def action_select(q, beta):
    softmax_sum = sum(exp(beta * v) for v in q)

    r = random.random()
    cumulative = 0

    for i in range(len(q)):
        p = exp(beta * q[i]) / softmax_sum
        cumulative += p
        if cumulative >= r:
            return i

    return RIGHT


# ------------------ ROBOT CLASS ------------------

class DelayedGratificationRobot(Node):

    def __init__(self):
        super().__init__('delayed_gratification_robot')

        self.navigator = BasicNavigator()

        # Initial pose
        self.start_pose = self.navigator.getPoseStamped(
            [HOME_X, HOME_Y],
            TurtleBot4Directions.NORTH
        )

        self.navigator.setInitialPose(self.start_pose)

        self.get_logger().info("Waiting for Nav2...")
        self.navigator.waitUntilNav2Active()
        time.sleep(2.0)

        self.run_experiment()

    # ------------------ NAVIGATION ------------------

    def navigate_to(self, x, y, orientation=None, timeout_sec=60.0):

        self.navigator.clearAllCostmaps()

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)

        if orientation is None:
            goal.pose.orientation.w = 1.0
        else:
            goal.pose.orientation = orientation

        self.get_logger().info(f"[Nav] Going to ({x:.3f}, {y:.3f})")

        self.navigator.goToPose(goal)

        start = time.time()

        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)

            if time.time() - start > timeout_sec:
                self.get_logger().warn("[Nav] Timeout — canceling")
                self.navigator.cancelTask()
                return False

        result = self.navigator.getResult()

        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("[Nav] SUCCEEDED")
            return True
        elif result == TaskResult.FAILED:
            self.get_logger().warn("[Nav] FAILED")
            return False
        else:
            self.get_logger().warn("[Nav] CANCELED")
            return False

    def go_home(self):
        self.get_logger().info("Returning home...")
        return self.navigate_to(
            HOME_X,
            HOME_Y,
            self.start_pose.pose.orientation
        )

    def go_to_chamber(self, action):
        if action == LEFT:
            x, y = LEFT_CHAMBER
        else:
            x, y = RIGHT_CHAMBER

        return self.navigate_to(x, y)

    # ------------------ EXPERIMENT LOGIC ------------------

    def print_q_table(self):
        self.get_logger().info("Q Table:")
        for s in range(STATES):
            self.get_logger().info(
                f"State {s}: L={qTbl[s][LEFT]:.3f}, R={qTbl[s][RIGHT]:.3f}"
            )

    def run_trial(self, p_wait):

        global qTbl

        # Select random state
        r = random.random()
        if r < 0.25:
            state = EXPM_LR
        elif r < 0.50:
            state = EXPM_RL
        elif r < 0.75:
            state = CTRL_LR
        else:
            state = CTRL_RL

        # Step 1: Reset
        ok = self.go_home()
        if not ok:
            self.get_logger().error("Failed to go home — aborting trial")
            return 0.0, False, state, None

        time.sleep(1.0)

        # Step 2: Think
        self.get_logger().info("Thinking...")
        time.sleep(2.0)

        # Step 3: Decision
        q_thinking = copy.deepcopy(qTbl)

        if state == EXPM_LR:
            q_thinking[state][LEFT] *= p_wait
        elif state == EXPM_RL:
            q_thinking[state][RIGHT] *= p_wait

        act = action_select(q_thinking[state], BETA)

        # Step 4: Act
        ok = self.go_to_chamber(act)
        if not ok:
            self.get_logger().warn("Chamber navigation failed")


        self.get_logger().info("Arrived at chamber")
        time.sleep(1.0)

        # Step 5: Reward
        if state == EXPM_LR:
            reward = LIVE_RWD if act == LEFT else DEAD_RWD
        elif state == EXPM_RL:
            reward = DEAD_RWD if act == LEFT else LIVE_RWD
        elif state == CTRL_LR:
            reward = UNOBTAINABLE_RWD if act == LEFT else DEAD_RWD
        else:
            reward = DEAD_RWD if act == LEFT else UNOBTAINABLE_RWD

        # Step 6: Update
        qTbl[state][act] += ALPHA * (reward - qTbl[state][act])

        experimental = state < CTRL_LR

        return reward, experimental, state, act

    # ------------------ EXPERIMENT RUN ------------------

    def run_experiment(self):

        TRIALS = 5

        for _ in range(TRIALS):
            self.run_trial(1.0)

        delays = np.array([10, 20])
        DELAY_TRIALS = 3

        self.get_logger().info("delay\texpm\tctrl")

        for d in delays:

            exp_cnt = 0
            ctrl_cnt = 0
            exp_success = 0
            ctrl_success = 0

            for _ in range(DELAY_TRIALS):

                rwd, ex, _, _ = self.run_trial(prob_wait(d))

                if ex:
                    exp_cnt += 1
                    exp_success += int(rwd > DEAD_RWD)
                else:
                    ctrl_cnt += 1
                    ctrl_success += int(rwd > UNOBTAINABLE_RWD)

            exp_rate = (exp_success * 100 / exp_cnt) if exp_cnt else 0
            ctrl_rate = (ctrl_success * 100 / ctrl_cnt) if ctrl_cnt else 0

            self.get_logger().info(f"{d}\t{exp_rate:.2f}\t{ctrl_rate:.2f}")

        self.print_q_table()


# ------------------ MAIN ------------------

def main(args=None):
    rclpy.init(args=args)

    node = DelayedGratificationRobot()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
