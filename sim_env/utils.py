import importlib
import os
import glob

class RobotModelLoader:
    DEFAULT_MUJOCO_DIR = "mujoco"
    ROBOT_PATH_STR = "ROBOT_MODEL_PATH.xml"
    MESH_DIR_STR = "MESH_DIR"
    
    DEFAULT_MESH_DIR = "assets"
    DEFAULT_SCENE_PATH = "./mj_pin_wrapper/sim_env/scene.xml"
    DEFAULT_MODELS_PATH = "./robots"
    
    def __init__(self,
                 robot_name: str,
                 **kwargs
                 ) -> None:
        self.robot_name = robot_name
        
        # Optional arguments
        optional_args = {
            "mesh_dir" : RobotModelLoader.DEFAULT_MESH_DIR,
            "mj_scene_path" : RobotModelLoader.DEFAULT_SCENE_PATH,
            "models_path" : RobotModelLoader.DEFAULT_MODELS_PATH,
        }
        optional_args.update(kwargs)
        for k, v in optional_args.items(): setattr(self, k, v)
        

        # Try from custom local files
        self.path_urdf, self.path_mjcf = "", ""
        # robot directory
        self.robot_dir = self._get_robot_dir()
        if self.robot_dir != "":
            # urdf description
            self.path_urdf = self._find_local_urdf()
            # xml description
            self.path_mjcf = self._find_local_xml()
            # package dir
            self.package_dir = self._get_local_package_dir()
        else:
            print(f"Robot model directory {self.robot_name} not found.")
            
        # Or try load from official git repo
        if (self.path_mjcf == "" or
            self.path_urdf == ""
            ):
            self.path_urdf,\
            self.path_mjcf,\
            self.package_dir = self._get_robot_model_paths_from_repo(self.robot_name)

        assert self.path_urdf != "", f"Robot urdf description not found."
        assert self.path_mjcf != "", f"Robot mjcf description not found."

    @staticmethod
    def get_paths(robot_name: str, **kwargs):
        loader = RobotModelLoader(robot_name, **kwargs)
        
        return loader.path_urdf, loader._get_xml_string(), loader.package_dir
    
    def _get_robot_dir(self):
        for dir_name in os.listdir(self.models_path):
            if self.robot_name in dir_name:
                robot_dir = os.path.join(self.models_path, dir_name)
                return robot_dir
        return ""
    
    def _find_local_urdf(self):
        urdf_files = glob.glob(self.robot_dir + "/*/*.urdf", recursive=True)
        if len(urdf_files) > 0:
            return urdf_files[0]
        return ""
        
    def _find_local_xml(self):
        # First look for mujoco dir
        for dir_name in os.listdir(self.models_path):
            if self.models_path in dir_name:
                mujoco_dir = os.path.join(self.models_path, dir_name)
                xml_files = glob.glob(mujoco_dir + "/*/*.xml", recursive=True)
                if len(xml_files) > 0:
                    return xml_files[0]
        
        # Otherwise look for .xml files
        xml_files = glob.glob(self.robot_dir + "/*/*.xml", recursive=True)
        if len(xml_files) > 0:
            return xml_files[0]
        return ""
        
    @staticmethod
    def _get_robot_model_paths_from_repo(robot_name:str):
        path_urdf = ""
        path_mjcf = ""
        package_dir = []
        
        try:
            robot_description_module = importlib.import_module(f"robot_descriptions.{robot_name}_description")
            robot_mj_description_module = importlib.import_module(f"robot_descriptions.{robot_name}_mj_description")

            path_urdf = robot_description_module.URDF_PATH
            path_mjcf = robot_mj_description_module.MJCF_PATH
            package_dir = [
                robot_description_module.PACKAGE_PATH,
                robot_description_module.REPOSITORY_PATH,
                os.path.dirname(robot_description_module.PACKAGE_PATH),
                os.path.dirname(robot_description_module.REPOSITORY_PATH),
                os.path.dirname(robot_description_module.URDF_PATH), 
            ]
            
        except Exception as e:
            print(e)
            print(f"Model description files of {robot_name} not found.")

        return path_urdf, path_mjcf, package_dir
    
    def _get_abs_path_mesh_dir(self) -> None:
        """
        Return mesh dir absolute path.
        """
        robot_model_dir = os.path.split(self.path_mjcf)[0]
        abs_path_mesh_dir = os.path.join(robot_model_dir, self.mesh_dir)
        return abs_path_mesh_dir
    
    def _get_local_package_dir(self) -> str:
        return [self.robot_dir] + glob.glob(self.robot_dir + "/*")
    
    def _get_xml_string(self) -> str:
        with open(self.mj_scene_path, 'r') as file:
            lines = file.readlines()
        
        # Find <mujoco> balise
        for i, line in enumerate(lines):
            if "<mujoco" in line:
                break
            
        # Add <include> line after mujoco balise
        include_str = f"""   <include file="{self.path_mjcf}"/>"""
        lines.insert(i + 1, include_str)
        xml_string = ''.join(lines)
                
        # Change mesh dir
        if 'meshdir=""' in xml_string:
            abs_path_mesh_dir = self._get_abs_path_mesh_dir()
            xml_string = xml_string.replace('meshdir=""',
                                            f'meshdir="{abs_path_mesh_dir}"')
            
        return xml_string
