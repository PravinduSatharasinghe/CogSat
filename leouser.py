import turtle


class LeoUserConfig:
    # Combined user coordinates for the single active LEO system
    LEO_USER_COORDINATES = [
        (-5, -5), (0, -75), (-75, 0), (-120, 150), (-150, 200),
        (-120, -150), (-150, -200), (75, -150), (150, -200), (-179, -5),
        (-103, -77), (145, 89), (4, -195), (-112, 148), (64, 169),
        (-144, -177), (-44, 33), (169, 137), (-45, 68), (-44, -17), (-127, -106),
        (5, 5), (0, 75), (75, 0), (-250, 0), (-150, 50),
        (120, 50), (175, 75), (120, 150), (175, 175),
        (278, -187), (-154, -37), (-87, 24), (222, 277),
        (103, 65), (-192, -134), (-196, 254), (272, -16),
        (92, -39), (116, 76), (-220, 132), (-226, -188)
    ]

    def __init__(self, l1_all_turtles, turtle_provider=turtle.Turtle):
        self.TurtleClass = turtle_provider
        self.l1_all_turtles = l1_all_turtles

        self.leo_user = []
        self.leo1_for_user = {}

        self.initialize_users()

    def initialize_users(self):
        for _ in self.LEO_USER_COORDINATES:
            leo_user_turtle = self.TurtleClass()
            self.leo_user.append(leo_user_turtle)

        for coordinates, user_turtle in zip(self.LEO_USER_COORDINATES, self.leo_user):
            self.configure_turtle(user_turtle, *coordinates)

        self.assign_users()

    def configure_turtle(self, turtle_obj, x, y):
        turtle_obj.penup()
        turtle_obj.shape("triangle")
        turtle_obj.fillcolor("Cyan")
        turtle_obj.speed("fastest")
        turtle_obj.goto(x, y)
        turtle_obj.pendown()

    def assign_users(self):
        """
        Assign each user to the nearest beam of the single LEO satellite.
        """
        self.leo1_for_user = {}

        for user_turtle in self.leo_user:
            nearest_beam = None
            nearest_distance = float("inf")

            for beam in self.l1_all_turtles:
                d = user_turtle.distance(beam)
                if d < nearest_distance:
                    nearest_distance = d
                    nearest_beam = beam

            self.leo1_for_user[user_turtle] = nearest_beam