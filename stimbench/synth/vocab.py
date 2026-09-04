from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple

CLASSES = ("ArmFlapping", "HeadBanging", "Spinning", "Normal")
STIMMING = CLASSES[:3]

LABELS = {c: c.lower() for c in CLASSES}

# cycles per second
TARGET_HZ = {"ArmFlapping": 3.0, "HeadBanging": 2.5, "Spinning": 1.0}

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


LIMP = ("the hands hanging limp from the wrists, flopping passively with all the "
        "strength gone out of them, trailing behind the forearms on every beat")
STILL = 'the shoulders and trunk staying still, the neck doing all the work'

TOPOGRAPHIES = {
    "ArmFlapping": [
        _t("af_forearms",
           "standing with both elbows bent and held a little away from the ribs, "
           "flapping both forearms and hands up and down together in unison, the "
           "upper arms swinging a little with them, " + LIMP,
           "standing"),
        _t("af_fingers",
           "holding both hands up near the chest and fluttering and flicking the "
           "fingers, the wrists bending and straightening repeatedly, the same "
           "finger pattern cycling over and over",
           "standing"),
        _t("af_shoulders",
           "holding both arms out to the sides and waving them repeatedly up and "
           "down from the shoulders, both arms mirroring each other exactly, "
           + LIMP,
           "standing"),
        _t("af_seated",
           "seated on the ground with the back straight, both arms lifted to "
           "chest height with the elbows flexed, shaking the forearms up and "
           "down in a repeated bilateral pattern, the legs settled and loose, "
           + LIMP,
           "seated"),
        _t("af_ragdoll",
           "standing with the arms hanging by the sides and the elbows soft, "
           "flapping both forearms from the elbows at waist height, "
           + LIMP + ", floppy like a rag doll's",
           "standing"),
        _t("af_excited",
           "standing and bouncing a little on the spot, both arms flapping up "
           "and down from the elbows, flapping harder as the excitement rises, "
           + LIMP,
           "standing"),
    ],
    # Rocking without head contact is a different stereotypy, so every variant
    # names a contact surface; the wall variant keeps the class out of no room.
    "HeadBanging": [
        _t("hb_sofa",
           "sitting on the floor with the back against the soft padded front of "
           "a sofa, knees drawn up, hands loose in the lap, throwing the back of "
           "the head hard against the upholstery and snapping it forward again, "
           "a sharp thump each time, " + STILL,
           "seated", {"sofa"}),
        _t("hb_chair",
           "sitting on a chair with the back against the backrest, throwing the "
           "back of the head hard against the top of the backrest and snapping "
           "it forward again, a sharp knock each time, the hands gripping the "
           "seat, " + STILL,
           "seated", {"chair"}),
        _t("hb_bed",
           "sitting up in bed with the back against the headboard and a thick "
           "pillow, throwing the back of the head hard into the pillow and "
           "snapping it forward again, a sharp thump each time, the arms loose "
           "at the sides, " + STILL,
           "seated", {"bed"}),
        _t("hb_thrash",
           "sitting on the floor with nothing behind, throwing the head forward "
           "and down sharply and snapping it back up, like headbanging to loud "
           "music, the hair whipping with it, the same quick violent nod over "
           "and over, " + STILL,
           "seated"),
        _t("hb_cushion",
           "kneeling on a padded play mat in front of a large soft cushion, "
           "slamming the forehead down onto the cushion and snapping it back "
           "up, a sharp thump each time, hands flat on the mat, " + STILL,
           "kneeling", {"mat"}),
        _t("hb_wall",
           "kneeling upright facing a bare stretch of wall, knocking the forehead "
           "hard against the wall and snapping it back, a sharp knock each time, "
           "the hands resting on the thighs, " + STILL,
           "kneeling", {"wall"}),
        _t("hb_floor",
           "kneeling on hands and knees, knocking the forehead hard down against "
           "the floor and snapping it back up, a sharp thump each time, the "
           "hands staying planted, " + STILL,
           "kneeling"),
    ],
    "Spinning": [
        _t("sp_shuffle",
           "standing in the middle of the open space and turning the whole body "
           "about a vertical axis, always in the same direction, feet taking small "
           "shuffling steps round in a circle",
           "rotating"),
        _t("sp_twirl",
           "twirling on the spot, the feet crossing over each other in small "
           "steps, the body rotating in one direction",
           "rotating"),
        _t("sp_arm",
           "spinning in place with one arm raised and the other held out to the "
           "side, the torso rotating together with the hips and legs",
           "rotating"),
    ],
    # Continuous activities only; the last four are goal-directed hard negatives
    # that superficially resemble a stimming class.
    "Normal": [
        _t("nm_blocks",
           "sitting cross legged on the ground stacking coloured wooden blocks "
           "into a tower, reaching for each block in turn, leaning forward to "
           "place it carefully and reaching back for the next",
           "seated"),
        _t("nm_crayon",
           "kneeling at a low table drawing on a sheet of paper with a crayon, "
           "head bent over the drawing, one hand steadying the paper, changing "
           "crayons once partway through and carrying on drawing",
           "kneeling", {"table"}),
        _t("nm_sofa_book",
           "sitting on the floor with the back resting against the front of a "
           "sofa, a picture book open on the knees, turning the pages one after "
           "another and looking down at them",
           "seated", {"sofa"}),
        _t("nm_bed_toy",
           "sitting up in bed against a thick pillow, holding a soft toy up in "
           "both hands and turning it over and over, looking at it and talking "
           "to it",
           "seated", {"bed"}),
        _t("nm_carry",
           "walking slowly across the space carrying a soft toy, weaving around "
           "a cushion on the ground and circling back the other way, looking "
           "down at the toy and up at the room in turn",
           "standing"),
        _t("nm_eat",
           "sitting at a table eating from a bowl with a spoon, lifting the "
           "spoon and lowering it, looking down at the food between mouthfuls",
           "seated", {"table"}),
        _t("nm_book",
           "turning the pages of a large picture book one after another, "
           "tracing a finger across each page and looking down at it",
           "seated"),
        _t("nm_tv",
           "sitting on the floor watching a cartoon on a television across the "
           "room, laughing and pointing at the screen, shuffling closer to see "
           "better",
           "seated"),
        _t("nm_ball",
           "walking a few steps, bending down to pick up a ball, standing back "
           "up with it and walking on, then bending down again for a second ball",
           "standing"),
        _t("nm_car",
           "kneeling on the floor pushing a toy car along the ground and "
           "following it around in loops, reaching ahead to steer it",
           "kneeling"),
        _t("nm_dance",
           "dancing to music, swaying from side to side and stepping in place, "
           "the arms swinging in big loose loops, each move different from the "
           "last",
           "standing"),
        _t("nm_wave",
           "standing and waving one hand overhead in a goodbye wave aimed at "
           "someone out of shot, lowering the hand, turning to look elsewhere "
           "and then lifting it to wave again as they come back into view",
           "standing", goal_directed=True),
        _t("nm_look",
           "turning around to look at something behind, then walking a few "
           "steps toward it and bending down to look at it closely",
           "standing", goal_directed=True),
        _t("nm_nod",
           "sitting on a chair at a table nodding along and talking to someone "
           "out of shot, gesturing with the hands, then picking up a cup from "
           "the table and drinking from it",
           "seated", {"chair", "table"}, goal_directed=True),
        _t("nm_clap",
           "clapping hands together in an uneven excited rhythm while watching "
           "someone out of shot, then reaching to pick up a toy from the table "
           "and holding it up to show them",
           "standing", {"table"}, goal_directed=True),
    ],
}


# Amplitude and body involvement only, never speed: clinical scales rate intensity
# apart from frequency. Keyed by posture so a seated child is never asked to lift
# the heels. Weighted toward the middle, like the overt cases that reach public video.
SEVERITY_LEVELS = ("subtle", "moderate", "pronounced", "whole_body")
SEVERITY_WEIGHTS = {"subtle": 0.25, "moderate": 0.25, "pronounced": 0.25,
                    "whole_body": 0.25}

SEVERITY = {
    ("ArmFlapping", "standing"): {
        "subtle": "the excursion small, the hands travelling only a few "
                  "centimetres and staying close in to the body, the shoulders "
                  "low, the rest of the body loose and easy, simple to overlook",
        "moderate": "the excursion moderate, the forearms covering a clear arc "
                    "beside the body, the shoulders and trunk carried along only "
                    "slightly",
        "pronounced": "the excursion wide, the forearms covering their full arc, "
                      "the shoulders raised and tight, the face tense",
        "whole_body": "the whole body involved, the arms covering their fullest "
                      "range while the trunk tenses and the heels lift clear of "
                      "the ground, the whole body shaking with it",
    },
    ("ArmFlapping", "seated"): {
        "subtle": "the excursion small, the hands travelling only a few "
                  "centimetres and staying close in to the body, the shoulders "
                  "low, the rest of the body loose and easy, simple to overlook",
        "moderate": "the excursion moderate, the forearms covering a clear arc "
                    "in front of the body, the shoulders and trunk carried along "
                    "only slightly",
        "pronounced": "the excursion wide, the forearms covering their full arc, "
                      "the shoulders raised and tight, the face tense",
        "whole_body": "the whole body involved, the arms covering their fullest "
                      "range while the trunk rocks and the legs bounce against "
                      "the ground with it, the whole body shaking",
    },
    ("HeadBanging", "seated"): {
        "subtle": "the head moving only a short distance each knock, quick small "
                  "jerks, the shoulders still",
        "moderate": "the head moving a clear and visible distance each knock, "
                    "the shoulders still",
        "pronounced": "the head thrown through a wide arc each knock, the "
                      "excursion unmistakable, the shoulders barely moving",
        "whole_body": "the head thrown as far as it will go each knock, the "
                      "shoulders jolting with each impact while the trunk stays "
                      "put, hair flying with it",
    },
    ("HeadBanging", "kneeling"): {
        "subtle": "the head moving only a short distance each knock, quick small "
                  "jerks, the shoulders still",
        "moderate": "the head moving a clear and visible distance each knock, "
                    "the shoulders still",
        "pronounced": "the head thrown through a wide arc each knock, the "
                      "excursion unmistakable, the shoulders barely moving",
        "whole_body": "the head thrown as far as it will go each knock, the "
                      "shoulders jolting with each impact while the trunk stays "
                      "put, hair flying with it",
    },
    ("Spinning", "rotating"): {
        "subtle": "the turn tight and small, the feet moving only a little, the "
                  "arms held in close to the body",
        "moderate": "the turn even and contained, the arms carried loosely away "
                    "from the body, balance easily kept",
        "pronounced": "the turn wide, the arms stretched right out, hair and "
                      "loose clothing carried outward, the child leaning into it",
        "whole_body": "the whole body committed, the arms flung to their fullest "
                      "extent, the head tipped back, the trunk leaning outward "
                      "against the turn, balance only just held",
    },
    # Same axis as activity level, so motion energy alone cannot separate Normal.
    ("Normal", "standing"): {
        "subtle": "the movements small and economical, the arms kept close to "
                  "the body and any steps short",
        "moderate": "the movements ordinary in size, an everyday amount of "
                    "reaching and stepping",
        "pronounced": "the movements large and expansive, reaching well out from "
                      "the body, the steps long and lively",
        "whole_body": "the whole body involved in every action, wide reaching "
                      "movements, the trunk and legs carried along with every "
                      "step",
    },
    ("Normal", "seated"): {
        "subtle": "the movements small and economical, the hands staying close "
                  "to the body",
        "moderate": "the movements ordinary in size, an everyday amount of "
                    "reaching and shifting",
        "pronounced": "the movements large, reaching well out from the body and "
                      "leaning to each side",
        "whole_body": "the whole upper body involved in every action, reaching "
                      "far out to both sides, the trunk twisting and leaning, "
                      "the position shifting often",
    },
}
SEVERITY[("Normal", "kneeling")] = SEVERITY[("Normal", "seated")]

# a topography whose amplitude lives in a different body part than its posture text
SEVERITY_BY_TOPOGRAPHY = {
    "af_fingers": {
        "subtle": "the fingers moving only slightly, the hands staying almost "
                  "still at the chest, simple to overlook",
        "moderate": "the fingers flicking clearly, the wrists bending a little "
                    "with them, the rest of the body loose",
        "pronounced": "the fingers and wrists flicking hard, the forearms joining "
                      "in, the shoulders raised and tight, the face tense",
        "whole_body": "the whole body involved, the hands flicking hard while "
                      "the trunk tenses and the heels lift clear of the ground, "
                      "the whole body shaking with it",
    },
}

# Clinical definition of a stereotypy in positive wording: video models drop negations.
STEREOTYPY_QUALIFIER = {
    "ArmFlapping": "caught up in excitement, the eyes fixed on what is exciting "
                   "them, the arms flapping on their own, the same pattern "
                   "repeating again and again",
    "HeadBanging": "the gaze drifting and unfocused, the movement carried on for "
                   "its own sake, the same pattern repeating again and again",
    "Spinning": "the gaze drifting and unfocused, the movement carried on for its "
                "own sake, the same pattern repeating again and again",
}

# what the child is reacting to; drawn for ArmFlapping only, since excitement
# is what sets flapping off in the real clips. Normal has its own screen activity.
AF_TRIGGER = [
    "",
    "",
    " while watching a cartoon on a television across the room",
    " while watching a toy train run round its track on the floor",
    " as a favourite song starts playing from somewhere out of shot",
    " while looking out of the window at something going on outside",
    " while watching bubbles drift past",
    " as someone comes in through the door",
]


# Rendered after "the child is"; nothing here may stop, pause or change tempo.
SECONDARY = {
    "ArmFlapping": [
        "shifting weight from one foot to the other as the movement continues",
        "glancing around and then away again without breaking the rhythm",
        "drifting a step or two across the ground and carrying on",
        "turning the head to one side and back while the arms keep going",
        "rocking gently forward and back at the same time",
        "swaying a little at the trunk, the weight shifting all the time",
        "grinning and squealing with excitement as the arms keep going",
        "bouncing up and down on the toes in time with the arms",
        "bobbing slightly at the knees in time with the arms",
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
        "reaching one hand out as if to steady against the air while turning",
        "turning the head to look around while the body keeps rotating",
        "letting the hair and clothing trail outward with the turn",
        "brushing hair back from the face mid turn",
        "looking up and then down again while turning",
        "letting the free hand open and close while turning",
        "breathing visibly, the mouth slightly open",
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
    "hb_cushion": _HANDS_PLANTED,
    "hb_floor": _HANDS_PLANTED | {"resettling the body against the surface between cycles"},
    "hb_wall": {"shifting the hands in the lap and gripping at the fabric",
                "resettling the body against the surface between cycles"},
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
    "amateur home video, low resolution and grainy, plain and unstaged, candid "
    "everyday footage, flat unflattering lighting, ordinary untidy "
    "surroundings, nothing composed or artistic about the shot",
    "casual family footage, slightly blurry with washed out colours, "
    "completely unedited, an ordinary everyday scene, plain and unremarkable, "
    "the kind of clip a parent records without thinking about it",
    "low quality home video uploaded to the internet years ago, compressed and "
    "soft, a candid unposed everyday moment, no styling and no framing care",
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
    "stiff, rigid, mannequin, shop dummy, doll, statue, robotic, wooden, "
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
