import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Transform, Vector3, Quaternion, Pose
import open3d as o3d
import numpy as np
from nav_msgs.msg import Odometry
import carla
import time
import threading
import datetime
import re

class PointCloudVisualizer:
    def __init__(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        # 获取世界和ego车辆
        world = client.get_world()
        time.sleep(1.0)
        target_vehicle_id = world.get_actors().filter('vehicle.audi.a2')[0].id
        self.node_name = 'carla_point_cloud_visualizer'
        self.topic_name = '/carla/agent_0/lidar'
        self.recording = False
        self.ego_location = [0, 0, 0]
        self.target_location = [0, 0, 0]
        self.target_vehicle = world.get_actor(target_vehicle_id) 

        self.point_cloud = None
        self.point_cloud_lock = threading.Lock()


        rospy.init_node(self.node_name, anonymous=True)

        self.transform_pub1 = rospy.Publisher('/carla/agent_0/control/set_transform', Pose, queue_size=10)
        self.transform_pub2 = rospy.Publisher('/carla/goal/control/set_transform', Pose, queue_size=10)

        rospy.Subscriber(self.topic_name, PointCloud2, self.lidar_callback)
        rospy.Subscriber('/carla/agent_0/odometry', Odometry, self.odometry_callback)


    def lidar_callback(self, data):
        # Convert ROS PointCloud2 message to Open3D PointCloud for visualization
        points = []
        for p in pc2.read_points(data, skip_nans=True):
            points.append([p[0], p[1], p[2]])
        
        self.point_cloud = np.array(points)
        # print("pointcloud updated, shape: ", np.array(points).shape)
    
    def odometry_callback(self, data):
        if not self.recording and data.twist.twist.linear.x != 0:
            self.recording = True  # 当odometry中的linear.x不为0时，设置标志位，开始记录
            print("Start pointcloud recording")
        pos_x = data.pose.pose.position.x
        pos_y = data.pose.pose.position.y
        pos_z = data.pose.pose.position.z
        self.ego_location = [pos_x, pos_y, pos_z]
        # print("ego_location: ", self.ego_location)
    
    def get_target_vehicle_bbox(self):
        bounding_box = self.target_vehicle.bounding_box
        transform = self.target_vehicle.get_transform()

        location = transform.location
        rotation = transform.rotation

        self.target_location = [location.x, -location.y, location.z]
        
        # 计算包围盒的8个角点
        bbox_corners = np.array([
            [bounding_box.extent.x, bounding_box.extent.y, bounding_box.extent.z],
            [-bounding_box.extent.x, bounding_box.extent.y, bounding_box.extent.z],
            [-bounding_box.extent.x, -bounding_box.extent.y, bounding_box.extent.z],
            [bounding_box.extent.x, -bounding_box.extent.y, bounding_box.extent.z],
            [bounding_box.extent.x, bounding_box.extent.y, -bounding_box.extent.z],
            [-bounding_box.extent.x, bounding_box.extent.y, -bounding_box.extent.z],
            [-bounding_box.extent.x, -bounding_box.extent.y, -bounding_box.extent.z],
            [bounding_box.extent.x, -bounding_box.extent.y, -bounding_box.extent.z]
        ])

        # bbox_corners膨胀一丢丢
        bbox_corners *= 1.1

        # 构造旋转矩阵
        rotation_matrix = np.array([
            [np.cos(np.deg2rad(rotation.yaw)), -np.sin(np.deg2rad(rotation.yaw)), 0],
            [np.sin(np.deg2rad(rotation.yaw)), np.cos(np.deg2rad(rotation.yaw)), 0],
            [0, 0, -1]
        ])

        bbox_corners = np.dot(bbox_corners, rotation_matrix.T)

        aa = self.ego_location
        bbox_corners += np.array([location.x - aa[0], location.y + aa[1], location.z + 1.0])

        r = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1]
        ])
        self.bbox_corners = np.dot(bbox_corners, r.T)
        # print(self.bbox_corners)

        return self.bbox_corners

    def get_points_in_bbox(self):
        min_coords = np.min(self.bbox_corners, axis=0)
        max_coords = np.max(self.bbox_corners, axis=0)

        while self.point_cloud is None:
            rospy.sleep(0.1)
            print("Waiting for point cloud data...")
        
        with self.point_cloud_lock:
            point_cloud_copy = self.point_cloud.copy()
        
        # print("Point cloud shape: ", point_cloud_copy.shape)

        points_inside_bbox = np.where((point_cloud_copy[:, 0] >= min_coords[0]) &
            (point_cloud_copy[:, 0] <= max_coords[0]) &
            (point_cloud_copy[:, 1] >= min_coords[1]) &
            (point_cloud_copy[:, 1] <= max_coords[1]) &
            (point_cloud_copy[:, 2] >= min_coords[2]) &
            (point_cloud_copy[:, 2] <= max_coords[2]))
        
        self.points_in_bbx = points_inside_bbox[0]

        # print("Points inside bbox: ", points_inside_bbox[0].shape[0])
    
        return points_inside_bbox[0]
    
    def create_bbox_lines(self, bbox_corners):
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        colors = [[1, 0, 0] for _ in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox_corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    def read_positions(self,filepath):
        poslist_1 = []
        poslist_2 = []

        with open(filepath, 'r') as file:
            for line in file:

                matches = re.findall(r'\[([^\]]+)\]', line)
                if len(matches) == 2:
                    coord1 = list(map(float, matches[0].split(',')))
                    coord2 = list(map(float, matches[1].split(',')))
                poslist_1.append(coord1)
                poslist_2.append(coord2)
             


        return poslist_1,poslist_2

    def set_transform(self, translation, rotation, publisher):


        pose = Pose()
        pose.position.x = translation[0]
        pose.position.y = translation[1]
        pose.position.z = translation[2]+0.5
        pose.orientation.x = rotation[0]
        pose.orientation.y = rotation[1]
        pose.orientation.z = rotation[2]
        pose.orientation.w = rotation[3]
       

        
        # Publish the transform
        # while not rospy.is_shutdown():
        publisher.publish(pose)
        rospy.loginfo_throttle(1,"Transform published: translation=%s, rotation=%s", translation, rotation)

    def euler_to_quaternion(self):
        # Function to convert Euler angles to quaternion
        qx = 1.6403277105699754e-06
        qy = -4.193950801044848e-05
        qz = 0.7071067328643186
        qw =  0.707106828263125


        
        return [qx,qy,qz,qw]

    def run(self):
        filepath = '/home/invs/new_ws/src/rda_ros/src/occ_ratio_2024-07-29_16-02-34.txt'
        poslist_1,poslist_2 = self.read_positions(filepath)
        yaw_radians_1 = 90 * np.pi / 180
        rotation_1 = self.euler_to_quaternion()
        rotation_2 = self.euler_to_quaternion()

        filename = '/home/invs/new_ws/src/rda_ros/src/occ729fenmu1.txt'


        for i in range(len(poslist_1)):


            self.set_transform(poslist_1[i], rotation_1, self.transform_pub1)
            self.set_transform(poslist_2[i], rotation_2, self.transform_pub2)
            rospy.sleep(2.0)


            bbx = self.get_target_vehicle_bbox()
            points = self.get_points_in_bbox()
            bbox_lines = self.create_bbox_lines(bbx)

            if self.recording:
                with open (filename, 'a') as f:
                    f.write(f"{self.ego_location},{self.target_location},{points.shape[0]} \n")




        
        rospy.spin()

if __name__ == '__main__':
    point_cloud_visualizer = PointCloudVisualizer()
    point_cloud_visualizer.run()