import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
from nav_msgs.msg import Odometry
import carla
import time
import threading
import datetime

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
    



    def run(self):

        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = '/home/invs/new_ws/src/rda_ros/src/occ_ratio_{}.txt'.format(current_time)

        while not rospy.is_shutdown():
            

            bbx = self.get_target_vehicle_bbox()
            points = self.get_points_in_bbox()
            bbox_lines = self.create_bbox_lines(bbx)

            if self.recording:
                with open (filename, 'a') as f:
                    f.write(f"{self.ego_location},{self.target_location},{points.shape[0]} \n")





            # point_cloud_o3d_1 = o3d.geometry.PointCloud()
            # point_cloud_o3d_1.points = o3d.utility.Vector3dVector(self.point_cloud)

            # o3d.visualization.draw_geometries([point_cloud_o3d_1, bbox_lines])

        
        rospy.spin()

if __name__ == '__main__':
    point_cloud_visualizer = PointCloudVisualizer()
    point_cloud_visualizer.run()