import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator

import random
from math import exp
import numpy as np
from scipy.stats import norm
import time
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions

"""
Kary Zheng
02 Feb 2026
Current version (testing)
"""

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

# ------------------ CHAMBER COORDINATES ------------------

LEFT_CHAMBER  = (1.9318245649,  0.5751032829)
RIGHT_CHAMBER = (1.9457954168, -0.2844115793)

# ------------------ Q TABLE ------------------

qTbl = [[0.0 for y in range(ACTIONS)] for x in range(STATES)]

# ------------------ FUNCTIONS (UNCHANGED LOGIC) ------------------

def prob_wait(tim):
    mean = 70
    std_dev = 20
    beta_weight = 2
    pw = 1 / np.exp(beta_weight * norm.cdf(tim, loc=mean, scale=std_dev))
    return pw


def action_select(q, beta):
    sumSoftMax = 0
    for i in range(len(q)):
        sumSoftMax += exp(beta*q[i])

    r = random.random()
    sumP = 0

    for i in range(len(q)):
        p = exp(beta*q[i]) / sumSoftMax
        sumP += p
        if sumP >= r:
            return i
    return RIGHT


# ------------------ ROBOT CLASS ------------------

class DelayedGratificationRobot(Node):
    def __init__(self):
        super().__init__('delayed_gratification_robot')

        self.navigator = BasicNavigator()

        # Build start pose FIRST
        self.start_pose = self.navigator.getPoseStamped(
            [-0.023635001853108406, -0.023635001853108406],
            TurtleBot4Directions.NORTH
        )

        # Set initial pose before Nav2 starts
        self.navigator.setInitialPose(self.start_pose)

        self.get_logger().info("Waiting for Nav2 to become active...")
        self.navigator.waitUntilNav2Active()
        time.sleep(2.0)

        self.run_experiment()

    # =========== Added & Modified here ===========
    def navigate_to(self, x, y, orientation=None, timeout_sec=60.0):
        self.navigator.cancelTask()
        time.sleep(0.2)
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
                self.get_logger().warn(f"Navigation timeout after {timeout_sec}s, canceling task")
                self.navigator.cancelTask()
                return False

        result = self.navigator.getResult()
        if result == BasicNavigator.TaskResult.SUCCEEDED:
            self.get_logger().info("Navigation SUCCEEDED")
            return True
        elif result == BasicNavigator.TaskResult.FAILED:
            self.get_logger().warn("Navigation FAILED")
            return False
        else:
            self.get_logger().warn("Navigation CANCELED")
            return False


    def go_to_pose_xy(self, x, y):
        return self.navigate_to(x, y)


    def go_home(self):
        self.get_logger().info("Returning to home pose...")

        x = self.start_pose.pose.position.x
        y = self.start_pose.pose.position.y
        orientation = self.start_pose.pose.orientation

        return self.navigate_to(x, y, orientation)



    def go_to_chamber(self, action):
        if action == LEFT:
            x, y = LEFT_CHAMBER
        else:
            x, y = RIGHT_CHAMBER

        return self.go_to_pose_xy(x, y)

    # ========================================================


# =============== Modified here =========================
# Since we would like to have the robot go back to the start points at the start of each trial, the running logic is:
# 1. Reset: Before each trial, it goes back to the START_POSE
# 2. Think: Once it goes back to the starting point, it pauses for a few seconds (as the thinking/decision time)
# 3. Act: Once the thinking is over, it goes to LEFT or RIGHT
    def print_q_table(self):
        self.get_logger().info("Current Q-Table:")
        for s in range(STATES):
            self.get_logger().info(f"State {s}: LEFT={qTbl[s][LEFT]:.3f}, RIGHT={qTbl[s][RIGHT]:.3f}")

    def run_trial(self, p_wait, delay_time=0):
        global qTbl

        r = random.random()
        if r < 0.25:
            current_state = EXPM_LR
        elif r < 0.50:
            current_state = EXPM_RL
        elif r < 0.75:
            current_state = CTRL_LR
        else:
            current_state = CTRL_RL

        # ========== Added here ============
        # 1. Trial start, go to the start point first
        self.get_logger().info("Going home before trial")
        ok = self.go_home()
        if not ok:
            self.get_logger().warn("go_home failed — retrying once...")
            self.go_home()


        # 2. Thinking
        # I'll just add a pause here to monitor that the robot is making decisions.
        self.get_logger().info("Thinking / Making  Decisions...")
        time.sleep(2.0)

        # =============================================

        # Modified
        import copy

        q_thinking = copy.deepcopy(qTbl)    # Changed here, with copy as a tempurary thinking rather than touching or changing the long-term memory 
        #(just in case if this time the robot doesn't want to wait, it forgets there are good shrimps there forever)

        # Apply patience ONLY to live shrimp choices
        if current_state == EXPM_LR:
            q_thinking[current_state][LEFT] *= p_wait
        elif current_state == EXPM_RL:
            q_thinking[current_state][RIGHT] *= p_wait

        act = action_select(q_thinking[current_state], BETA)

        # Robot physically goes to chamber
        self.go_to_chamber(act)
        self.get_logger().info("Arrived at chamber, consuming reward...")
        time.sleep(1.0)

        # Determine reward EXACTLY like your simulation
        if current_state == EXPM_LR:
            rwd = LIVE_RWD if act == LEFT else DEAD_RWD
        elif current_state == EXPM_RL:
            rwd = DEAD_RWD if act == LEFT else LIVE_RWD
        elif current_state == CTRL_LR:
            rwd = UNOBTAINABLE_RWD if act == LEFT else DEAD_RWD
        else:
            rwd = DEAD_RWD if act == LEFT else UNOBTAINABLE_RWD

        # Q update (same)
        qTbl[current_state][act] = qTbl[current_state][act] + ALPHA*(rwd - qTbl[current_state][act])

        experimental_trial = current_state < CTRL_LR
        return rwd, experimental_trial, current_state, act


    def run_experiment(self):
        
        # ---------- TRAINING PHASE ----------
        #TRIALS = 100
        TRIALS=5
        for t in range(TRIALS):
            self.run_trial(1.0, delay_time=0)

        # ---------- DELAY TEST PHASE ----------
        # delays = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130])
        delays = np.array([10,20])
        # DELAY_TRIALS = 100
        DELAY_TRIALS=3
        delay_gratification_experiment_results = np.zeros(len(delays))
        delay_gratification_ctrl_results = np.zeros(len(delays))

        exp_trials = np.zeros(len(delays))
        ctrl_trials = np.zeros(len(delays))
        self.get_logger().info("delay\texpm\tctrl")

        for d in range(len(delays)):
            exp_cnt = 0
            ctrl_cnt = 0

            for t in range(DELAY_TRIALS):
                rwd, ex, state, act = self.run_trial(prob_wait(delays[d]), delay_time=delays[d])

                if ex:
                    exp_cnt += 1
                    exp_trials[d] += 1
                    delay_gratification_experiment_results[d] += int(rwd > DEAD_RWD)
                else:
                    ctrl_cnt += 1
                    ctrl_trials[d] += 1
                    delay_gratification_ctrl_results[d] += int(rwd > UNOBTAINABLE_RWD)

            self.get_logger().info(
                f"{delays[d]}\t"
                f"{(delay_gratification_experiment_results[d]*100)/exp_cnt:.2f}\t"
                f"{(delay_gratification_ctrl_results[d]*100)/ctrl_cnt:.2f}"
            )
        self.get_logger().info("EXPERIMENT FINISHED, PRINTING Q TABLE")
        self.print_q_table()

# ------------------ MAIN ------------------

def main(args=None):
    rclpy.init(args=args)
    node = DelayedGratificationRobot()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()