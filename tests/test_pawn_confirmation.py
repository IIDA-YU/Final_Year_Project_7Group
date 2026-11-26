import unittest

from algorithm.pawn_confirmation import PawnConfirmation
from camera.pawn_detection import CameraPose, DEFAULT_INTRINSICS, PawnCamera


class PawnConfirmationTests(unittest.TestCase):
    def test_confirms_after_four_seconds_of_stationary_samples(self):
        confirmation = PawnConfirmation()
        self.assertEqual(confirmation.observe([(120, 10)], 0, 0), ())
        self.assertTrue(confirmation.waiting)
        confirmation.stopped(0.1)
        self.assertEqual(confirmation.observe([(105, 9)], 0.2, 0.5), ())

        for tick in range(1, 9):
            time = tick * 0.5
            point = (155, 29) if tick == 3 else (135, 15)
            self.assertEqual(confirmation.observe([point], time, time), ())
        self.assertTrue(confirmation.waiting)
        self.assertEqual(confirmation.observe([(137, 16)], 4.5, 4.5), ((135, 15),))
        self.assertFalse(confirmation.waiting)

    def test_intermittent_false_positives_allow_progress_and_a_real_recheck(self):
        confirmation = PawnConfirmation()
        point = [(140, 0)]
        confirmation.observe(point, 0, 0)
        confirmation.stopped(0)
        confirmation.observe(point, 0.3, 0.3)
        self.assertEqual(confirmation.observe([], 0.5, 0.5), ())
        self.assertFalse(confirmation.waiting)

        for tick in range(6, 46):
            time = tick / 10
            confirmation.observe(point if tick % 2 == 0 else [], time, time)
            self.assertFalse(confirmation.waiting)

        for time in (4.6, 4.8, 5.0, 5.2):
            confirmation.observe(point, time, time)
        self.assertFalse(confirmation.waiting)
        confirmation.observe(point, 5.4, 5.4)
        self.assertTrue(confirmation.waiting)

    def test_gap_and_timeout_break_confirmation(self):
        for observation_time in (1.5, None):
            with self.subTest(observation_time=observation_time):
                confirmation = PawnConfirmation()
                point = [(140, 0)]
                confirmation.observe(point, 0, 0)
                confirmation.stopped(0)
                confirmation.observe(point, 0.3, 0.3)
                if observation_time is None:
                    confirmation.expire(6.1)
                else:
                    self.assertEqual(confirmation.observe(point, observation_time, observation_time), ())
                self.assertFalse(confirmation.waiting)

    def test_camera_keeps_empty_results_between_detections(self):
        camera = PawnCamera()
        camera.set_enabled(True)
        pose = CameraPose((0, 0), 0, 0, (1, 2))
        box = (120, 100, 170, 155)
        for time, boxes in ((1, [box]), (2, []), (3, [box])):
            camera.publish(time, pose, boxes, (320, 240), DEFAULT_INTRINSICS)

        enabled, frames = camera.read_frames()
        self.assertTrue(enabled)
        self.assertEqual([frame.received_at for frame in frames], [1, 2, 3])
        self.assertEqual([len(frame.positions) for frame in frames], [1, 0, 1])
        self.assertEqual(camera.read_frames(), (True, ()))


if __name__ == "__main__":
    unittest.main()
