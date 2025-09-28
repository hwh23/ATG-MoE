from point_renderer.rvt_renderer import RVTBoxRenderer as BaseRVTBoxRenderer
from point_renderer.cameras import OrthographicCameras, PerspectiveCameras

class RVTBoxRenderer(BaseRVTBoxRenderer):
    def __init__(
        self,
        img_size,
        radius=0.012,
        default_color=0.0,
        default_depth=-1.0,
        antialiasing_factor=1,
        pers=False,
        normalize_output=True,
        with_depth=True,
        device="cuda",
        perf_timer=False,
        strict_input_device=True,
        no_down=True,
        no_top=False,
        three_views=False,
        two_views=False,
        one_view=False,
        add_3p=False,
        **kwargs):
        super().__init__(
            img_size=img_size,
            radius=radius,
            default_color=default_color,
            default_depth=default_depth,
            antialiasing_factor=antialiasing_factor,
            pers=pers,
            normalize_output=normalize_output,
            with_depth=with_depth,
            device=device,
            perf_timer=perf_timer,
            strict_input_device=strict_input_device,
            no_down=no_down,
            no_top=no_top,
            three_views=three_views,
            two_views=two_views,
            one_view=one_view,
            add_3p=add_3p,
            **kwargs)
    
    def _get_cube_cameras(
        self,
        img_size,
        orthographic,
        no_down,
        no_top,
        three_views,
        two_views,
        one_view,
        add_3p,
    ):
        cam_dict = {
            "top": {"eye": [0, 0, 1], "at": [0, 0, 0], "up": [0, 1, 0]},
            "front": {"eye": [1, 0, 0], "at": [0, 0, 0], "up": [0, 0, 1]},
            "down": {"eye": [0, 0, -1], "at": [0, 0, 0], "up": [0, 1, 0]},
            "back": {"eye": [-1, 0, 0], "at": [0, 0, 0], "up": [0, 0, 1]},
            "left": {"eye": [0, -1, 0], "at": [0, 0, 0], "up": [0, 0, 1]},
            "right": {"eye": [0, 0.5, 0], "at": [0, 0, 0], "up": [0, 0, 1]},
        }

        assert not (two_views and three_views)
        assert not (one_view and three_views)
        assert not (one_view and two_views)
        assert not add_3p, "Not supported with point renderer yet,"
        if two_views or three_views or one_view:
            if no_down or no_top or add_3p:
                print(
                    f"WARNING: when three_views={three_views} or two_views={two_views} -- "
                    f"no_down={no_down} no_top={no_top} add_3p={add_3p} does not matter."
                )

        if three_views:
            cam_names = ["top", "back", "left"] # original: ["top", "front", "right"] in rvt_renderer.py
        elif two_views:
            cam_names = ["top", "front"]
        elif one_view:
            cam_names = ["front"]
        else:
            cam_names = ["top", "front", "down", "back", "left", "right"]
            if no_down:
                # select index of "down" camera and remove it from the list
                del cam_names[cam_names.index("down")]
            if no_top:
                del cam_names[cam_names.index("top")]


        cam_list = [cam_dict[n] for n in cam_names]
        eyes = [c["eye"] for c in cam_list]
        ats = [c["at"] for c in cam_list]
        ups = [c["up"] for c in cam_list]

        if orthographic:
            # img_sizes_w specifies height and width dimensions of the image in world coordinates
            # [2, 2] means it will image coordinates from -1 to 1 in the camera frame
            cameras = OrthographicCameras.from_lookat(eyes, ats, ups, img_sizes_w=[2, 2], img_size_px=img_size)
        else:
            cameras = PerspectiveCameras.from_lookat(eyes, ats, ups, hfov=70, img_size=img_size)
        return cameras