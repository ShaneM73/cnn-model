import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths below are relative to training/
import json, numpy as np
M=json.load(open('model.json'))
R=np.load('tuning.npy'); R=R/R.max(1,keepdims=True)
M['tuning']=np.round(R,3).tolist()
L1=[("Falling diagonal \\","Peaks on a \\ stroke and goes almost silent on / ."),
("Upright stroke |","Prefers vertical strokes; nearly silent on flat ones."),
("Steep rising /","Prefers / strokes between 45° and vertical; silent on a shallow \\ ."),
("Flat or gently rising","Likes flat and shallow / strokes, so it lights up along the bottoms of boxes and triangles."),
("Anything but upright","Responds to flats and both diagonals, and drops out on near-vertical strokes. Busiest at the centre of an X."),
("Flat stroke, underside","Ink above and blank below: the underside of flat or falling strokes. Silent on a 60° / ."),
("Upright stroke |, again","A second vertical detector with slightly different weights. Small networks often learn near-duplicates."),
("Flat stroke, centred","Weights along its middle row: flat strokes and the tops of corners like ┐.")]
L2=[("Corner ┐","A flat stroke turning down at the right: top-right corners of boxes, and plus crossings."),
("Arch ∩","The top of a curve: a stroke rising and falling again."),
("Falling arc","A \\ stroke that bends as it goes, from circles and slanted sides."),
("Bulging side )","A stroke curving round on the right, the side of a ring."),
("Diagonal junctions","Places where diagonals cross or end: X centres, triangle corners."),
("Peak ∧","Two strokes meeting at a point above, like a triangle's apex."),
("Upright with crossbar","A vertical stroke, often with a flat one crossing it."),
("Upswing ╯","A stroke curving upward to the right, the lower right of a ring."),
("Corner ┐, sharp","A crisp right-angle turn from flat to downward."),
("Diagonal crossing ╳","A / and a \\ crossing: the middle of an X."),
("Inner corner ┘","A short upright meeting a flat stroke, seen from the inside of a bend."),
("Left bend (","A stroke curving round on the left, or a └ corner.")]
L3=[("X arm, upper right","The upper end of a falling diagonal arm, as found in an X."),
("Long falling diagonal","A long \\ stroke: the arm of an X or the right side of a triangle."),
("Low curve ‿","A shallow curve near the bottom: the base of a ring or a rounded corner."),
("Whole triangle △","Its 18×18 window holds most of a triangle: slanted sides meeting a base."),
("Corner └","A bottom-left right angle: an upright meeting a flat base."),
("Left corner or crossing","An upright and a flat stroke meeting on the left, or two diagonals crossing."),
("Plus crossing ┼","A short upright and flat stroke crossing at right angles."),
("Closed right side ⊐","A stroke that turns down and back on the right: a box's or a ring's right side."),
("X arm, top","The top of a falling diagonal, close to the edge of the drawing."),
("Slanted side meets base","A diagonal ending on a flat stroke, the bottom-right of a triangle."),
("Crossbar ┬","A flat stroke with an upright through or below its middle."),
("Arch ∩","A curving top, from rings, or the top half of an X."),
("Whole X","Two long diagonals crossing near the middle. Almost only fires for crosses."),
("Box top ┌┐","A flat top with an upright dropping from it: the top of a square."),
("Short upright tick","A short vertical stroke near the top: a plus's upper arm."),
("Falling diagonal, lower","The lower part of a \\ stroke, from crosses and triangles.")]
for lay,names in zip(M['layers'],(L1,L2,L3)):
    for c,(n,d) in zip(lay['channels'],names): c['name']=n; c['desc']=d
    lay['purity']=round(float(np.mean([max(c['top200']) for c in lay['channels']])),3)
print([l['purity'] for l in M['layers']], M['probe'])
json.dump(M,open('model.json','w'),separators=(',',':'),ensure_ascii=False)
