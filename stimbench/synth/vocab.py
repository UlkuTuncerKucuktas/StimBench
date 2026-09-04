from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple

CLASSES = ("ArmFlapping", "HeadBanging", "Spinning", "Normal")
STIMMING = CLASSES[:3]

LABELS = {c: c.lower() for c in CLASSES}

# cycles per second
# real-world rates, used for the repetition count in the prompt. Flapping renders at
# the requested rate; head banging renders near 1.1 Hz whatever is requested
# (13 clips, requests 1.5 to 4.0 Hz). Achieved period is measured after generation.
TARGET_HZ = {"ArmFlapping": 3.0, "HeadBanging": 2.5, "Spinning": 1.0}
# measured on smoke sets: flapping renders about 1.5x the count it is asked for, so
# the count in the prompt is scaled down; requested_hz stays the real target
COUNT_SCALE = {"ArmFlapping": 0.7}

WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
        7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
        12: "twelve"}


@dataclass(frozen=True)
class Topography:
    id: str
    text: str
    posture: str                               # standing | seated | rotating
    needs: FrozenSet[str] = frozenset()        # environment tags required
    goal_directed: bool = False                # Normal hard negatives


def _t(id, text, posture, needs=(), goal_directed=False):
    return Topography(id, text, posture, frozenset(needs), goal_directed)


LIMP = ("the hands hanging limp from the wrists like a puppet's hands on slack "
        "strings, fingers soft and slightly parted, each hand lagging a beat behind "
        "its forearm and whirling loosely round the wrist at the turn of every stroke")
KNOCK = ("each knock quick and hard, one after another like a rock fan headbanging to "
         "fast music, the shoulders and chest staying planted, only the head and neck "
         "moving")

TOPOGRAPHIES = {
    # one form, chosen against the real clips: arms hanging low and swinging big
    # from the elbows, limp hands whirling at the wrists, the feet staying put;
    # variants change only height, seat or walking
    "ArmFlapping": [
        _t("af_sides",
           "standing still on planted feet with both arms hanging down at the sides, "
           "elbows soft, the forearms swinging forward and back from the elbows while "
           "the limp hands whirl loosely at the wrists, " + LIMP,
           "standing"),
        _t("af_sides_high",
           "standing still on planted feet with both arms hanging down at the sides, "
           "elbows soft, the forearms swinging high forward and back from the elbows, "
           "mouth open in a grin, " + LIMP,
           "standing"),
        _t("af_hips",
           "standing with the arms hanging at the sides and the elbows slightly bent, "
           "the forearms flicking up and down from the elbows just beside the hips, "
           + LIMP,
           "standing"),
        _t("af_waist_front",
           "standing with the elbows soft and the forearms swinging a little forward "
           "and back in front of the waist, the upper arms staying close to the "
           "body, " + LIMP,
           "standing"),
        _t("af_seated_sides",
           "sitting cross-legged on the floor with both arms hanging loose at the "
           "sides, elbows soft, the forearms swinging a little forward and back from "
           "the elbows, " + LIMP,
           "seated"),
        _t("af_walking",
           "walking slowly across the room with both arms hanging down at the sides, "
           "elbows soft, the forearms swinging forward and back from the elbows, "
           + LIMP,
           "standing"),
    ],
    # neck-driven at every severity: the real clips keep the trunk still, and any
    # kneeling forward variant renders as a prostration
    "HeadBanging": [
        _t("hb_sofa",
           "sitting on the floor, back leaning against the padded front of a sofa, "
           "knees drawn up, hands resting in the lap, rocking the head fast back and "
           "forth from the neck so the back of the head slams into the cushion and "
           "rebounds forward each time, " + KNOCK,
           "seated", {"sofa"}),
        _t("hb_chair",
           "sitting upright on a chair with a padded backrest, back against it, hands "
           "gripping the seat edge, the head rocking hard back and forth from the "
           "neck so the back of the head slams into the backrest and rebounds, the "
           "hair bouncing, " + KNOCK,
           "seated", {"chair"}),
        _t("hb_bed",
           "sitting up in bed, back against a thick pillow on the headboard, arms "
           "loose at the sides, the head whipping back into the pillow and rebounding "
           "forward, fast and hard, the pillow denting on every hit, " + KNOCK,
           "seated", {"bed"}),
        _t("hb_thrash",
           "sitting cross-legged on the floor, hands resting on the knees, nodding "
           "the head fast and hard from the neck, chin driving down toward the chest "
           "and jerking back up, hair flying, like headbanging to loud fast music, "
           + KNOCK,
           "seated"),
        _t("hb_wall",
           "sitting on the floor with the back and shoulders flat against a bare "
           "wall, legs stretched out in front, hands resting on the thighs, rocking "
           "the head fast from the neck so the back of the head knocks hard against "
           "the wall and rebounds each time, " + KNOCK,
           "seated", {"wall"}),
        _t("hb_wall_standing",
           "standing close to a bare wall, both palms flat on the wall at shoulder "
           "height, feet planted, nodding the head fast and hard from the neck so the "
           "forehead knocks against the wall and rebounds each time, the hips still, "
           + KNOCK,
           "standing", {"wall"}),
    ],
    # "{dir}" is filled per clip with "to the left" or "to the right"; the view-change
    # clause keeps the subject turning instead of the camera orbiting
    "Spinning": [
        _t("sp_arms_out",
           "standing in the open middle of the floor and turning the whole body round "
           "on the spot {dir}, both arms held out to the sides at shoulder height, the "
           "feet taking small side steps in a tight circle, the back turning to the "
           "camera and the face coming round again",
           "rotating"),
        _t("sp_arms_down",
           "standing on the open floor and turning the whole body round on the spot "
           "{dir}, the arms hanging by the sides with the hands loose, the feet "
           "shuffling in small steps within one floor tile, the shoulders and head "
           "turning together so the back and then the face come round to the camera",
           "rotating"),
        _t("sp_one_arm",
           "turning the whole body round on the spot {dir} with one hand raised above "
           "the head and the other arm out at shoulder height, both arms held still "
           "against the trunk, the feet stepping in a tight circle, hips and shoulders "
           "turning as one",
           "rotating"),
        _t("sp_look_up",
           "turning the whole body round on the spot {dir} with the head tipped back "
           "and the eyes on the ceiling light, the arms out to the sides at hip height, "
           "the feet taking small steps in a tight circle, the back and then the face "
           "coming round to the camera",
           "rotating"),
    ],
    # Broad activity, caption register, the same rooms, postures and furniture
    # contact as the stimming classes; nothing rhythmic, looping or swaying, which
    # the RBS scores as a stereotypy. The last four are goal-directed hard negatives
    # whose goal is visible in the action.
    "Normal": [
        _t("nm_blocks",
           "sitting cross legged on the floor stacking wooden blocks into a tower, "
           "picking one block at a time from a pile at the side, leaning forward to "
           "set it on top, sitting back and reaching for the next",
           "seated"),
        _t("nm_crayon",
           "kneeling upright at a low table drawing on a sheet of paper with a crayon, "
           "the eyes on the paper, one hand flat on the page, swapping the crayon for "
           "another colour from a box and drawing again",
           "kneeling", {"table"}),
        _t("nm_puzzle",
           "kneeling at a low table fitting large puzzle pieces into a wooden board, "
           "picking a piece up, trying it in one slot, turning it and pressing it into "
           "place, then reaching for the next piece",
           "kneeling", {"table"}),
        _t("nm_sofa_tablet",
           "sitting on the floor with the back against the front of a sofa, a tablet "
           "held in both hands on the knees, swiping across the screen with one "
           "finger, tilting the tablet to look at it, tapping once, then lifting the "
           "head to glance across the room",
           "seated", {"sofa"}),
        _t("nm_tv",
           "sitting on the floor watching a cartoon on a television across the room, "
           "laughing and pointing at the screen once, then shuffling forward on the "
           "bottom to sit closer",
           "seated"),
        _t("nm_bed_toy",
           "sitting up in bed against a thick pillow holding a soft toy in both hands, "
           "turning it to look at its face, tucking it under the blanket, then pulling "
           "it out and setting it on the pillow beside them",
           "seated", {"bed"}),
        _t("nm_bed_brush",
           "sitting on the edge of the bed brushing the hair with a hairbrush, drawing "
           "the brush down from the crown, glancing into a small hand mirror held in "
           "the other hand, then brushing the other side",
           "seated", {"bed"}),
        _t("nm_eat",
           "sitting on a chair at a table eating from a bowl with a spoon, lifting the "
           "spoon to the mouth and lowering it to the bowl, picking up a cup to "
           "drink, looking down at the bowl between mouthfuls",
           "seated", {"chair", "table"}),
        _t("nm_book",
           "sitting on the floor with a large picture book open on the lap, turning a "
           "page, pointing at a picture with one finger and looking down at it, then "
           "turning the next page",
           "seated"),
        _t("nm_carry_box",
           "walking across the room carrying a soft toy in both arms to a toy box by "
           "the wall, bending to put it in, then walking back to pick up a second toy "
           "from the floor",
           "standing"),
        _t("nm_ball_basket",
           "walking a few steps, bending at the knees to pick up a ball from the floor, "
           "straightening up and dropping it into a basket, then walking to a second "
           "ball and picking that one up",
           "standing"),
        _t("nm_car",
           "kneeling on the floor pushing a toy car along the edge of the rug with one "
           "hand, crawling forward on the knees to follow it, turning the car round at "
           "the rug corner and pushing it back the other way",
           "kneeling"),
        _t("nm_wall_stickers",
           "standing close to the wall sticking paper stickers onto a large poster "
           "taped to the wall, peeling a sticker from a sheet held in one hand, "
           "pressing it flat with the palm, stepping back to look at the poster, then "
           "choosing a spot for the next one",
           "standing", {"wall"}),
        _t("nm_dance",
           "dancing to music from a speaker on the shelf, stepping to one side and "
           "back, bending the knees, lifting one arm and then the other, adding a hop, "
           "the hands staying below shoulder height",
           "standing"),
        _t("nm_bubbles",
           "standing on the lawn blowing soap bubbles from a wand, dipping the wand "
           "into a bottle held in the other hand, lifting it to the mouth and blowing, "
           "then reaching up to pop a bubble",
           "standing", {"outdoor"}),
        _t("nm_scooter",
           "standing on a kick scooter on a garden path, pushing off with one foot, "
           "gliding a short way, putting the foot down to stop, then turning the "
           "scooter round by hand and pushing off again",
           "standing", {"outdoor"}),
        _t("nm_wave",
           "standing and lifting one hand to wave goodbye toward a person in the "
           "doorway, the hand swinging twice above the shoulder, then lowering it, "
           "picking up a bag from the floor and walking toward the door",
           "standing", goal_directed=True),
        _t("nm_look_turn",
           "standing facing the camera, turning the whole body round once to look at "
           "a toy on the shelf behind, walking two steps to the shelf and lifting the "
           "toy down, then turning back and holding it up to look at it",
           "standing", goal_directed=True),
        _t("nm_balloon",
           "standing and bumping a balloon upward with the forehead, watching it fall, "
           "catching it in both hands, then tossing it up and bumping it once more",
           "standing", goal_directed=True),
        _t("nm_clap",
           "sitting on a chair at a table clapping the hands together three times "
           "toward a puppet show on a tablet propped on the table, then reaching out "
           "to pick up a block and holding it up toward the screen",
           "seated", {"chair", "table"}, goal_directed=True),
    ],
}


# Amplitude and body involvement only, never speed: clinical scales rate intensity
# apart from frequency. Keyed by posture so a seated child is never asked to lift
# the heels. Weighted toward the middle, like the overt cases that reach public video.
SEVERITY_LEVELS = ("subtle", "moderate", "pronounced", "whole_body")
SEVERITY_WEIGHTS = {"subtle": 0.25, "moderate": 0.25, "pronounced": 0.25,
                    "whole_body": 0.25}
# flapping under-renders at low amplitude (a subtle clip reads as Normal), so the
# class is weighted to the overt tiers by decision of the owner after viewing
SEVERITY_WEIGHTS_BY_CLASS = {
    "ArmFlapping": {"subtle": 0.05, "moderate": 0.05, "pronounced": 0.45, "whole_body": 0.45},
}
# share of a class given to each variant; unlisted variants split the remainder
TOPOGRAPHY_WEIGHTS = {
    "ArmFlapping": {"af_sides": 0.9},
}

SEVERITY = {
    ("ArmFlapping", "standing"): {
        "subtle": "the forearms travelling a short distance, the hands twirling "
                  "softly close to the body, the shoulders low and relaxed, feet still",
        "moderate": "the forearms swinging a clear distance from the elbows, the hands "
                    "whirling freely at the wrists, the shoulders low and relaxed, feet still",
        "pronounced": "the forearms swinging wide from the elbows up to chest height and "
                      "back down, the loose hands spinning round the wrists at each turn, "
                      "the arms in constant big motion, the feet still",
        "whole_body": "the whole arms swinging their fullest from the shoulders, up past "
                      "the chest and back behind the hips, the loose hands spinning round "
                      "the wrists at every turn, the trunk swaying with it, the feet still",
    },
    ("ArmFlapping", "seated"): {
        "subtle": "the forearms travelling a short distance, the hands twirling "
                  "softly close to the body, the shoulders low and relaxed",
        "moderate": "the forearms swinging a clear distance from the elbows, the hands "
                    "whirling freely at the wrists, the shoulders low and relaxed",
        "pronounced": "the forearms swinging wide from the elbows up to chest height and "
                      "back down, the loose hands spinning round the wrists at each turn, "
                      "the arms in constant big motion, the seat staying put",
        "whole_body": "the whole arms swinging their fullest from the shoulders, up past "
                      "the chest and back beside the seat, the loose hands spinning round "
                      "the wrists at every turn, the trunk rocking with it, the seat staying put",
    },
    ("HeadBanging", "seated"): {
        "subtle": "the head moving only a short distance, quick small knocks, the "
                  "shoulders still",
        "moderate": "the head swinging a clear distance each knock, hair bouncing, the "
                    "shoulders still",
        "pronounced": "the head swinging its full range each knock, hair flying, the "
                      "shoulders jolting on impact while the chest stays planted",
        "whole_body": "the head hurled its full range each knock, hair flying, the "
                      "shoulders jolting on each impact, the child grunting with the "
                      "effort, the chest and hips staying put",
    },
    ("HeadBanging", "standing"): {
        "subtle": "the head moving only a short distance, quick small knocks, the "
                  "shoulders still",
        "moderate": "the head swinging a clear distance each knock, hair bouncing, the "
                    "shoulders still",
        "pronounced": "the head swinging its full range each knock, hair flying, the "
                      "shoulders jolting on impact while the chest stays planted",
        "whole_body": "the head hurled its full range each knock, hair flying, the "
                      "shoulders jolting on each impact, the child grunting with the "
                      "effort, the chest and hips staying put",
    },
    ("Spinning", "rotating"): {
        "subtle": "the turn tight and small, each step a few centimetres, the trunk "
                  "upright, the head level, balance easy",
        "moderate": "the turn even and contained, the trunk upright, the head level with "
                    "the shoulders, the steps a little wider, balance easily kept",
        "pronounced": "the turn wide, the trunk tilted a little to one side, the head "
                      "tilted, the steps wide, the free hand swinging out with the turn",
        "whole_body": "the whole body committed to the turn, the trunk leaning, the head "
                      "tipped back, the arms carried outward by the turn, the steps long "
                      "and stumbling, balance only just held",
    },
    # Same axis as activity level, so motion energy alone cannot separate Normal.
    ("Normal", "standing"): {
        "subtle": "the movements small and economical, the arms staying near the body, "
                  "the steps short, the trunk upright",
        "moderate": "the movements ordinary in size, the arms swinging a little with "
                    "each step, the trunk turning as the task needs",
        "pronounced": "the movements large, the arms reaching well out from the body, "
                      "the steps long, the trunk bending and turning with each action",
        "whole_body": "the whole body in every action, the arms at full stretch, deep "
                      "knee bends, big steps and turns, the trunk leaning far forward "
                      "and back",
    },
    ("Normal", "seated"): {
        "subtle": "the hands moving close to the lap, the trunk still, the head bending "
                  "only a little toward the task",
        "moderate": "the hands moving freely in front of the body, the trunk leaning a "
                    "little toward the task",
        "pronounced": "reaching well out to the sides and forward, the trunk twisting "
                      "and leaning, the weight shifting on the seat",
        "whole_body": "the whole upper body in every action, reaching to full stretch on "
                      "both sides, the trunk twisting and leaning far forward, the "
                      "sitting position changing often",
    },
    ("Normal", "kneeling"): {
        "subtle": "sitting back on the heels, the hands moving close to the body, the "
                  "trunk still",
        "moderate": "kneeling upright, the hands moving freely in front, the trunk "
                    "leaning a little toward the task",
        "pronounced": "rising up on the knees to reach, the arms out to the sides and "
                      "forward, the trunk twisting and leaning",
        "whole_body": "the whole body in every action, rising high on the knees and "
                      "dropping back to the heels, reaching to full stretch, the trunk "
                      "leaning far forward and back, the knees shifting position often",
    },
}

# a topography whose amplitude lives in a different body part than its posture text
SEVERITY_BY_TOPOGRAPHY = {}

# Clinical definition of a stereotypy in positive wording: video models drop negations.
STEREOTYPY_QUALIFIER = {
    "ArmFlapping": "caught up in excitement, eyes locked on what is exciting them",
    "HeadBanging": "face slack, eyes half closed and unfocused, mouth slightly open, "
                   "the same beat repeating again and again",
    "Spinning": "the eyes turned to one side and unfocused, the face slack and calm, "
                "the mouth a little open, the gaze sliding past the room with each turn",
}

# what the child is reacting to; drawn for ArmFlapping only, since excitement
# is what sets flapping off in the real clips. Normal has its own screen activity.
AF_TRIGGER = [
    "",
    "",
    " while watching a cartoon on the television",
    " while watching a toy train on its track",
    " as a favourite song starts playing",
    " while looking out of the window",
    " as bubbles drift past",
    " as a parent walks in with a favourite toy",
]

# appended to the shared negative prompt for one class; invisible in the data,
# steers the render away from the look-alike each class collapses into
NEGATIVE_BY_CLASS = {
    "ArmFlapping": ", clapping, waving hello, hands clasped, jumping jacks, praying, "
                   "jumping, hopping, bouncing, feet leaving the floor",
    "HeadBanging": ", bowing, praying, prostration, kneeling, sujud, rocking the whole "
                   "body, nodding slowly",
    "Spinning": ", orbiting camera, rotating camera, camera spin, turntable, ballerina, "
                "pirouette, tutu, dance performance, spinning top",
    "Normal": ", rocking, swaying, hand flapping, head shaking, spinning on the spot, tremor",
}

DIRECTION = ("to the left", "to the right")
NORMAL_PACE = ("carrying out every action at an unhurried pace",
               "carrying out every action at a brisk pace",
               "carrying out the actions in quick bursts with short pauses between them")


# Rendered after "the child is"; nothing here may stop, pause or change tempo.
SECONDARY = {
    "ArmFlapping": [
        "shifting weight from one foot to the other as the movement continues",
        "glancing around and then away again without breaking the rhythm",
        "drifting a step or two across the ground and carrying on",
        "turning the head to one side and back while the arms keep going",
        "rocking gently forward and back at the same time",
        "swaying a little at the trunk, the weight shifting all the time",
        "grinning with excitement as the arms keep going",
        "leaning a little toward what is exciting them as the arms keep going",
        "the mouth opening wider as the arms keep going",
        "turning slowly on the spot to face a different way while continuing",
        "looking down and then up again as the movement goes on",
        "breathing visibly, the shoulders rising and falling with it",
    ],
    "HeadBanging": [
        "shifting the hands in the lap and gripping at the fabric",
        "bringing one hand up to touch the face briefly",
        "letting the eyes drift to one side and back",
        "shifting the legs, one knee dropping out to the side",
        "resettling the body against the surface between cycles",
        "blinking slowly, the gaze unfocused",
        "keeping the hands where they are, only the head moving",
    ],
    "Spinning": [
        "lifting one hand to push the hair back from the face mid turn",
        "the eyes closing for a moment and opening again mid turn",
        "one foot catching the other and a quick extra step to recover balance",
        "one hand opening and closing loosely at the side while turning",
        "a short laugh with the mouth open and the eyes screwed up",
        "the head tilting to one shoulder and straightening while the body turns",
        "a small sideways stagger after a turn, then the turn taken up again",
        "the fingertips of one hand brushing a piece of furniture once as the body passes it",
        "the mouth a little open, the chest rising and falling visibly",
    ],
    "Normal": [
        "glancing up and around between actions",
        "shifting position and resettling while carrying on",
        "fidgeting with clothing with the free hand",
        "looking toward something off to one side and back",
        "making small natural adjustments of balance and posture throughout",
        "shifting weight from one foot to the other",
        "brushing hair back from the face",
        "breathing visibly, the body loose and easy",
    ],
}

_SEATED_BLOCK = {"shifting weight from one foot to the other",
                 "shifting weight from one foot to the other as the movement "
                 "continues",
                 "drifting a step or two across the ground and carrying on",
                 "turning slowly on the spot to face a different way while "
                 "continuing",
                 "bobbing slightly at the knees in time with the arms"}
SECONDARY_POSTURE_BLOCK = {
    "standing": {"shifting the legs, one knee dropping out to the side",
                 "shifting the hands in the lap and gripping at the fabric"},
    "seated": _SEATED_BLOCK,
    "kneeling": _SEATED_BLOCK | {"shifting the legs, one knee dropping out to the side"},
    "rotating": set(),
}
_HANDS_PLANTED = {"shifting the hands in the lap and gripping at the fabric",
                  "bringing one hand up to touch the face briefly"}
SECONDARY_TOPOGRAPHY_BLOCK = {
    "hb_wall_standing": {"shifting the hands in the lap and gripping at the fabric",
                         "shifting the legs, one knee dropping out to the side",
                         "resettling the body against the surface between cycles"},
    "hb_chair": {"shifting the legs, one knee dropping out to the side"},
}
# a head that is knocking cannot also be held tilted or turned away
POSE_CLASS_BLOCK = {
    "HeadBanging": {"the head tilted over to one side", "the head turned away to one side",
                    "the chin tucked down toward the chest"},
}

SHORT_HAIR = ("buzz cut", "bowl cut", "short bob", "closely cropped",
              "short ginger hair", "short dark hair", "short blond hair")


AGE = ["a 3 year old", "a 4 year old", "a 5 year old", "a 6 year old",
       "a 7 year old", "an 8 year old", "a 9 year old", "a 10 year old"]

GENDER = ("boy", "girl")
GENDER_LABEL = {"boy": "M", "girl": "F"}

BUILD = ["slightly built", "sturdily built", "small for their age",
         "tall and lanky for their age", "of average build",
         "chubby cheeked and stocky"]

HAIR = {
    "boy": ["short dark hair", "short blond hair", "a short buzz cut",
            "messy light brown hair", "short tightly curled black hair",
            "straight black hair cut in a blunt fringe", "shaggy brown hair",
            "a bowl cut that needs trimming", "short ginger hair",
            "thick wavy dark hair", "closely cropped hair with a side part",
            "dark hair sticking up at the back from sleeping on it",
            "longish blond hair tucked behind the ears"],
    "girl": ["long brown hair tied in a ponytail",
             "long dark hair loose over the shoulders",
             "curly black hair in two bunches",
             "shoulder length blond hair", "black hair in a short bob",
             "long red hair in a plait", "short curly brown hair",
             "dark hair in two braids", "fine blond hair with a fringe",
             "thick black curly hair tied back off the face",
             "wavy brown hair with a plastic clip in it",
             "long straight black hair hanging loose",
             "light brown hair in a messy half-fallen ponytail"],
}

SKIN = ["light skin", "pale freckled skin", "fair skin with rosy cheeks",
        "medium olive skin", "light brown skin", "brown skin",
        "deep brown skin", "warm tan skin", "golden brown skin", "dark skin"]

# (text, tag): "home" needs a domestic room, "school" a school room and age >= 5.
CLOTHING = {
    "boy": [
        ("a faded red t-shirt and grey shorts", "any"),
        ("a striped long sleeved top and jeans", "any"),
        ("blue checked pyjamas", "home"),
        ("a grey tracksuit with white stripes", "any"),
        ("a green hoodie and dark joggers", "any"),
        ("a plain white t-shirt and denim shorts", "any"),
        ("a navy sweatshirt with a dinosaur print", "any"),
        ("a football shirt a size too big", "any"),
        ("a mustard jumper and corduroy trousers", "any"),
        ("a cartoon print t-shirt and tracksuit bottoms", "any"),
        ("a checked shirt worn open over a t-shirt", "any"),
        ("denim dungarees over a long sleeved top", "any"),
        ("a school polo shirt and grey trousers", "school"),
    ],
    "girl": [
        ("a pink cotton dress", "any"),
        ("purple leggings and a flowered top", "any"),
        ("yellow pyjamas with small stars", "home"),
        ("a yellow jumper and blue jeans", "any"),
        ("a striped long sleeved top and a denim skirt", "any"),
        ("a plain white t-shirt and denim shorts", "any"),
        ("a lilac tracksuit", "any"),
        ("a red tartan pinafore over a white top", "any"),
        ("a glittery unicorn t-shirt and patterned leggings", "any"),
        ("a knitted cardigan over a plain cotton dress", "any"),
        ("denim dungarees over a long sleeved top", "any"),
        ("an oversized t-shirt and bright leggings", "any"),
        ("a school polo shirt and a grey pleated skirt", "school"),
    ],
}

# Generated children default to catalogue-clean. (text, tag): indoor-only items tagged.
DETAIL = [
    ("", "any"), ("", "any"),
    (", barefoot", "indoor"),
    (", in odd mismatched socks", "indoor"),
    (", in socks with no shoes on", "indoor"),
    (", with a plaster on one knee", "any"),
    (", with a faint food stain down the front", "any"),
    (", the sleeves slightly too long and half covering the hands", "any"),
    (", hair a little messy and unbrushed", "any"),
    (", wearing small round glasses", "any"),
    (", with a sticker stuck on the back of one hand", "any"),
    (", one trouser leg rucked up above the sock", "trousers"),
    (", the top on slightly askew with the collar twisted", "any"),
]


# Tags: indoor|outdoor, domestic|institutional, a family (home|school|nursery|clinic|hall),
# and the furniture present (sofa|bed|mat|table|chair|wall). Light is never described here.
@dataclass(frozen=True)
class Environment:
    id: str
    text: str
    tags: FrozenSet[str]

    @property
    def family(self):
        for f in ("home", "school", "nursery", "clinic", "hall", "outdoor"):
            if f in self.tags:
                return f
        return "other"


def _e(id, text, *tags):
    return Environment(id, text, frozenset(tags))


ENVIRONMENTS = [
    _e("living_room",
       "in a cluttered family living room, a worn beige sofa with a crumpled "
       "throw blanket over one arm, a patterned rug over laminate flooring, a "
       "low coffee table holding two mugs and a remote control, framed "
       "photographs on a magnolia wall",
       "sofa", "table", "wall", "indoor", "domestic", "home"),
    _e("child_bedroom",
       "in a small child's bedroom, a single bed with a rumpled duvet against "
       "the wall, a low white wardrobe with one door ajar, stickers stuck on "
       "the wardrobe, a plastic toy box overflowing in the corner",
       "bed", "wall", "indoor", "domestic", "home"),
    _e("shared_bedroom",
       "in a bedroom shared by two children, a wooden bunk bed with both "
       "duvets unmade, posters taped up slightly crooked, clothes dropped on "
       "the floor, a nightlight plugged in by the skirting board",
       "bed", "wall", "indoor", "domestic", "home"),
    _e("kitchen",
       "in a narrow family kitchen, pale wooden cupboards and a cluttered "
       "worktop with a kettle and a fruit bowl, a fridge covered in magnets "
       "and children's drawings, a small table with two chairs against the "
       "wall, tiled floor",
       "table", "chair", "wall", "indoor", "domestic", "home"),
    _e("dining_room",
       "in a small dining room, a wooden dining table pushed against the wall "
       "with mismatched chairs around it, a sideboard covered in papers and "
       "post, scuffed laminate flooring",
       "table", "chair", "wall", "indoor", "domestic", "home"),
    _e("classroom",
       "in a plain primary school classroom, small tables pushed together with "
       "little blue chairs around them, a low bookshelf against the wall, "
       "children's paintings pinned to a display board, scuffed vinyl floor",
       "table", "chair", "wall", "indoor", "institutional", "school"),
    _e("corridor",
       "in a school corridor, coat pegs at child height with bags and coats "
       "hanging from them, a painted line running along the wall, notice "
       "boards covered in leaflets, shiny lino floor",
       "wall", "indoor", "institutional", "school"),
    _e("playroom",
       "in a carpeted playroom, stackable plastic toy crates along one wall, a "
       "foam alphabet mat in the corner, a beanbag chair, crayon marks on the "
       "skirting board",
       "mat", "chair", "wall", "indoor", "domestic", "home"),
    _e("nursery",
       "in a nursery play area, low soft foam blocks stacked to one side, a "
       "mirror panel fixed along one wall at child height, primary coloured "
       "storage units, a padded play mat laid out in the middle of the soft "
       "foam flooring",
       "mat", "wall", "indoor", "institutional", "nursery"),
    _e("garage_playroom",
       "in a converted garage used as a playroom, bare plasterboard walls half "
       "painted, an old rug and a thick foam play mat thrown over bare "
       "concrete, storage boxes stacked against one side",
       "mat", "wall", "indoor", "domestic", "home"),
    _e("hallway",
       "in a narrow hallway with a coat rack overloaded with jackets, shoes "
       "kicked into a pile by the front door, a radiator under a small window, "
       "worn carpet with a threadbare patch",
       "wall", "indoor", "domestic", "home"),
    _e("garden",
       "in a small suburban back garden, patchy grass and a low wooden fence, a "
       "plastic slide off to one side, a washing line with clothes pegged on it, "
       "the brick back wall of the house on one side",
       "wall", "outdoor"),
    _e("garden_mat",
       "in a back garden on a padded outdoor play mat spread over the grass, a "
       "large floor cushion at one end of it, a plastic paddling pool "
       "deflated nearby, a brick garden wall behind",
       "mat", "wall", "outdoor"),
    _e("balcony",
       "on the tiled balcony of a flat, a folding drying rack with clothes on "
       "it, plant pots along the railing, the rendered wall of the flat on one "
       "side, other apartment blocks visible behind",
       "wall", "outdoor"),
    _e("playground",
       "in a public playground, rubber safety surfacing underfoot, a climbing "
       "frame and swings behind, a bench to one side, a low painted wall along "
       "one edge, railings and bare trees beyond",
       "wall", "outdoor"),
    _e("soft_play",
       "in a soft play centre, padded coloured foam shapes and a ball pit edge "
       "behind, a padded play mat underfoot, safety netting along one side",
       "mat", "indoor", "institutional", "hall"),
    _e("therapy_room",
       "in a bare therapy room, a blue foam mat on the floor, a single low "
       "table with a box of sensory toys, plain off white walls, a wall clock "
       "and a small window high up",
       "mat", "table", "wall", "indoor", "institutional", "clinic"),
    _e("waiting_area",
       "in a clinic waiting area, rows of linked plastic chairs against the "
       "wall, a low table with dog-eared magazines, a noticeboard covered in "
       "leaflets, hard vinyl flooring",
       "chair", "table", "wall", "indoor", "institutional", "clinic"),
    _e("flat_living_room",
       "in a cramped living room in a flat, a two seater sofa against the "
       "window, a drying rack with laundry on it, a bookcase crammed with "
       "folders",
       "sofa", "wall", "indoor", "domestic", "home"),
    _e("grandparents",
       "in a grandparents' front room, a dark patterned carpet, a floral "
       "armchair beside an older sofa, a lace doily and photographs on a "
       "sideboard, heavy curtains and a gas fire",
       "sofa", "chair", "wall", "indoor", "domestic", "home"),
    _e("apartment",
       "in an apartment living room, a large patterned rug laid over tiled "
       "flooring, a low couch with cushions along the wall, a glass tea table, "
       "net curtains at the window",
       "sofa", "table", "wall", "indoor", "domestic", "home"),
    _e("tidying",
       "in a living room half way through being tidied, sofa cushions pulled "
       "off onto the floor, a vacuum cleaner left standing in the middle of the "
       "room, cardboard boxes part unpacked",
       "sofa", "wall", "indoor", "domestic", "home"),
    _e("church_hall",
       "in a church hall used for a playgroup, stacked chairs against the wall, "
       "a large soft play mat in the middle of a wooden floor, high windows, "
       "painted brick walls",
       "mat", "chair", "wall", "indoor", "institutional", "hall"),
]

# Never names furniture, so it cannot reference a sofa the room lacks.
CLUTTER = {
    "domestic": [
        "the floor scattered with toys, odd socks and torn bits of paper",
        "a pile of unfolded laundry spilling out of a basket onto the floor",
        "cushions pulled off and dumped in a heap, a blanket trailing across "
        "the floor",
        "plastic toys, a tipped over cup and crumbs trodden underfoot",
        "cardboard boxes and carrier bags stacked untidily against the wall",
        "children's drawings and stickers stuck unevenly all over the lower "
        "half of the wall",
        "a knocked over toy box with its contents spread right across the floor",
        "shoes, bags and coats dropped in a heap by the door",
        "half eaten snacks, a juice carton and torn packaging left out",
        "books and jigsaw pieces spread everywhere with nothing put away",
        "a duvet and a pillow dragged down onto the floor and left in a heap",
        "washing draped over a folding clothes horse, clutter on every surface",
        "toy cars, building blocks and a deflated balloon strewn about",
        "a stack of dirty plates and loose paper left out on every surface",
    ],
    "institutional": [
        "toys and activity mats left out where they were last used",
        "a stack of plastic crates pushed to one side, the lids not matching",
        "leaflets and paper spilled across a low surface",
        "coats and small bags dropped short of their pegs",
        "chairs pushed out at odd angles and not put back",
        "scuffed paintwork and marks along the wall at child height",
        "a box of toys tipped over near the wall, the contents spread out",
        "loose paper, crayons and a dropped beaker left on the floor",
    ],
    "outdoor": [
        "toys and a deflated ball left out on the ground",
        "a knocked over bucket and spade and chalk scattered on the ground",
        "washing hanging on a line and garden things left lying about",
        "a scattering of leaves, a dropped coat and an upturned plastic crate",
        "muddy footprints, a bike on its side and toys left where they fell",
        "plant pots, a watering can and small shoes left out on the ground",
    ],
}

# Body attitude only; viewpoint belongs to the camera slot.
POSE = {
    "standing": [
        "the head tilted over to one side",
        "the chin tucked down toward the chest",
        "the shoulders hunched up toward the ears",
        "the gaze directed upward and past everything",
        "the eyes half closed",
        "the mouth slightly open and the jaw slack",
        "up on the toes with the heels lifted clear",
        "one shoulder carried lower than the other",
        "the knees slightly bent and the body carried low",
        "the head turned away to one side",
    ],
    "seated": [
        "the head tilted over to one side",
        "the chin tucked down toward the chest",
        "the shoulders hunched up toward the ears",
        "the eyes half closed",
        "the mouth slightly open and the jaw slack",
        "the back rounded and the spine curved",
        "one knee dropped out to the side",
        "the gaze directed down and unfocused",
        "the head turned away to one side",
    ],
    "kneeling": [
        "the head tilted over to one side",
        "the chin tucked down toward the chest",
        "the shoulders hunched up toward the ears",
        "the eyes half closed",
        "the mouth slightly open and the jaw slack",
        "the back rounded and the spine curved",
        "the gaze directed down and unfocused",
        "the head turned away to one side",
    ],
    "rotating": [
        "the head carried upright",
        "the chin lifted",
        "the eyes half closed",
        "the mouth slightly open",
        "the head tipped back a little",
        "the shoulders loose and dropped",
        "the gaze unfocused and sweeping",
    ],
}


@dataclass(frozen=True)
class Camera:
    id: str
    text: str
    aspect: str          # portrait | landscape
    motion: str          # handheld | static


CAMERA = [
    Camera("hand_wobble",
           "unsteady handheld framing that drifts and wobbles slightly, the "
           "subject sliding off centre and being corrected", "portrait", "handheld"),
    Camera("fixed_low",
           "a completely fixed viewpoint from low down, the framing never moving "
           "for the whole clip", "portrait", "static"),
    Camera("fixed_high_back",
           "a fixed viewpoint from well back and set slightly too high, looking "
           "down a little at the scene", "landscape", "static"),
    Camera("hand_sway",
           "loose handheld framing that sways gently and occasionally reframes "
           "to follow the movement", "landscape", "handheld"),
    Camera("tilted_vertical",
           "slightly tilted vertical framing with the subject off centre and too "
           "much empty space above the head", "portrait", "static"),
    Camera("adult_height",
           "a static viewpoint at adult standing height, framing a little wide "
           "and off centre", "landscape", "static"),
    Camera("ground_level",
           "a very low viewpoint down at ground level looking slightly upward at "
           "the child", "portrait", "static"),
    Camera("overhead",
           "a high viewpoint looking steeply down from above adult head height",
           "landscape", "static"),
    Camera("hand_nudge",
           "a shot that starts badly framed and is nudged to recentre the child "
           "partway through", "landscape", "handheld"),
    Camera("oblique",
           "a viewpoint from well off to one side, the child seen at a sharp "
           "oblique angle", "landscape", "static"),
    Camera("off_axis",
           "an off axis viewpoint round to one side rather than square on to the "
           "child", "landscape", "static"),
    Camera("tilted_horizon",
           "a tilted viewpoint with the horizon line noticeably not level",
           "landscape", "static"),
    Camera("seated_height",
           "a viewpoint at seated height looking across at the child",
           "landscape", "static"),
    Camera("obstructed",
           "a framing partly obstructed along one edge by something dark in the "
           "foreground", "landscape", "static"),
]

SHOT = [
    "wide framing showing the whole body from head to feet with the background "
    "visible",
    "medium framing from the knees up, the child filling about half the height "
    "of the frame",
    "medium close framing from the waist up, the head and arms clearly visible",
    "fairly wide framing with a lot of the untidy surroundings visible around "
    "the child",
    "medium framing with the child well off to one side of the frame",
]

LIGHT = {
    "indoor": [
        "lit by flat grey daylight through a window on one side, the opposite "
        "side of the face in shadow, exposure slightly too bright",
        "lit by a single dim yellow ceiling bulb, warm colour cast, the corners "
        "falling into shadow",
        "lit by a table lamp in the evening, orange pooled light on one side "
        "and a dark unlit background",
        "lit by harsh bluish fluorescent ceiling tubes, flat shadowless light, "
        "slightly green colour cast",
        "backlit by a bright window so the child is underexposed and a little "
        "dark against a blown out background",
        "lit by mixed daylight and a warm indoor bulb, inconsistent white "
        "balance across the frame",
        "lit by low orange late afternoon sun coming in through a window, long "
        "shadows stretched across the floor",
        "lit by dull overcast daylight from a window only, flat grey and "
        "slightly underexposed",
        "lit by a bright ceiling light against a dark window, hard shadows "
        "directly below the child",
    ],
    "outdoor": [
        "lit by flat overcast daylight, grey and slightly underexposed",
        "lit by bright direct sunlight with hard dark shadows on the ground",
        "lit by low orange late afternoon sun casting long shadows",
        "lit by dull grey daylight under a heavy sky, colours muted and flat",
        "lit by dappled sunlight coming through trees, patches of light and "
        "shade shifting across the scene",
        "lit by bright hazy daylight, the sky blown out white behind",
    ],
}

# Nobody here is described as motionless; that would fight the negative prompt.
EXTRA = {
    "indoor": [
        "", "", "", "", "", "",
        ", a television on in the background showing a bright cartoon",
        ", a cat curled up on a chair in the background",
        ", a birthday banner still taped to the wall from an earlier party",
        ", a folded pushchair leaning against the wall",
        ", a pair of small shoes left in the middle of the floor",
        ", a radio playing somewhere out of shot",
        ", a window open with a curtain moving in the draught",
        ", another child sitting in the background looking at a book",
        ", an adult partly visible at the very edge of the frame, only an arm "
        "and shoulder in shot",
        ", a parent sitting nearby in the background looking down at something "
        "in their lap",
        ", two other children playing together in the far background",
    ],
    "outdoor": [
        "", "", "", "", "",
        ", a bicycle lying on its side nearby",
        ", washing moving on a line in the background",
        ", a ball rolling to a stop across the ground",
        ", birds moving about in the background",
        ", a folded pushchair leaning against a fence",
        ", another child playing further off in the background",
        ", an adult partly visible at the very edge of the frame, only an arm "
        "and shoulder in shot",
        ", two other children running about in the far background",
    ],
}

PEOPLE_MARKERS = ("child", "adult", "parent")

# Pulls the model off its cinematic default without naming a recording device.
AESTHETIC = [
    "amateur home video, low resolution and grainy, candid and unstaged, flat "
    "unflattering lighting, ordinary untidy surroundings",
    "casual family footage, slightly blurry with washed out colours, unedited, "
    "an ordinary everyday scene, plain and unremarkable",
    "low quality home video uploaded years ago, compressed and soft, a candid "
    "unposed everyday moment, no styling",
]

# "still"/"motionless" are absent on purpose: severity texts ask for stillness,
# and guidance must not be pushed both ways. Frozen video is "static image".
NEGATIVE = (
    "nude, nudity, naked, undressed, underwear, bare chest, nsfw, sexual, "
    "cinematic, film still, professional lighting, shallow depth of field, "
    "bokeh, colour graded, teal and orange, dramatic, slow motion, timelapse, "
    "studio, staged, model, posed, "
    "phone, smartphone, mobile phone, camera, camcorder, webcam, tripod, "
    "screen, display, tablet, hand holding a phone, person filming, "
    "someone recording, recording device, selfie, mirror reflection, "
    "cartoon, anime, illustration, 3d render, cgi, video game, watermark, "
    "text, subtitles, logo, split screen, static image, frozen frame, "
    "clenched fists, fists, stiff hands, rigid wrists, "
    "stiff, rigid, mannequin, shop dummy, statue, robotic, wooden, "
    "only one part of the body moving, distorted limbs, extra limbs, "
    "extra arms, deformed hands, fused fingers, melting body, warping face, "
    "blurry mess"
)


def validate():
    for cls in CLASSES:
        assert cls in TOPOGRAPHIES and TOPOGRAPHIES[cls], cls
        for t in TOPOGRAPHIES[cls]:
            assert (cls, t.posture) in SEVERITY, (cls, t.posture)
            assert t.posture in POSE, t.posture
            for tag in t.needs:
                assert any(tag in e.tags for e in ENVIRONMENTS), (t.id, tag)
    for cls in CLASSES:
        for e in ENVIRONMENTS:
            assert any(t.needs <= e.tags for t in TOPOGRAPHIES[cls]), (cls, e.id)
    for key, levels in SEVERITY.items():
        assert tuple(levels) == SEVERITY_LEVELS, key
    assert abs(sum(SEVERITY_WEIGHTS.values()) - 1.0) < 1e-9
    ids = [e.id for e in ENVIRONMENTS]
    assert len(ids) == len(set(ids)), "duplicate environment id"
    ids = [c.id for c in CAMERA]
    assert len(ids) == len(set(ids)), "duplicate camera id"
    for cls, tops in TOPOGRAPHIES.items():
        ids = [t.id for t in tops]
        assert len(ids) == len(set(ids)), f"duplicate topography id in {cls}"


validate()
