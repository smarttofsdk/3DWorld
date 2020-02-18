from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
from shapely.validation import explain_validity
import pyclipper


def _coords(shape):
    """
    Return a list of lists of coordinates of the polygon. The list consists
    firstly of the list of exterior coordinates followed by zero or more lists
    of any interior coordinates.
    """

    assert shape.geom_type == 'Polygon'
    coords = [list(shape.exterior.coords)]
    for interior in shape.interiors:
        coords.append(list(interior.coords))
    return coords


def _drop_degenerate_inners(shape):
    """
    Drop degenerate (zero-size) inners from the polygon.

    This is implemented as dropping anything with a size less than 0.5, as the
    polygon is in integer coordinates and the smallest valid inner would be a
    triangle with height and width 1.
    """

    assert shape.geom_type == 'Polygon'

    new_inners = []
    for inner in shape.interiors:
        # need to make a polygon of the linearring to get the _filled_ area of
        # the closed ring.
        if abs(Polygon(inner).area) >= 0.5:
            new_inners.append(inner)

    return Polygon(shape.exterior, new_inners)


def _contour_to_poly(contour):
    poly = Polygon(contour)
    if not poly.is_valid:
        poly = poly.buffer(0)
    assert poly.is_valid, \
        "Contour %r did not make valid polygon %s because %s" \
        % (contour, poly.wkt, explain_validity(poly))
    return poly


def _polytree_node_to_shapely(node):
    """
    Recurses down a Clipper PolyTree, extracting the results as Shapely
    objects.

    Returns a tuple of (list of polygons, list of children)
    """

    polygons = []
    children = []
    for ch in node.Childs:
        p, c = _polytree_node_to_shapely(ch)
        polygons.extend(p)
        children.extend(c)

    if node.IsHole:
        # check expectations: a node should be a hole, _or_ return children.
        # this is because children of holes must be outers, and should be on
        # the polygons list.
        assert len(children) == 0
        if node.Contour:
            children = [node.Contour]
        else:
            children = []

    elif node.Contour:
        poly = _contour_to_poly(node.Contour)
        for ch in children:
            inner = _contour_to_poly(ch)
            diff = poly.difference(inner)
            if not diff.is_valid:
                diff = diff.buffer(0)

            # keep this for when https://trac.osgeo.org/geos/ticket/789 is
            # resolved.
            #
            #  assert diff.is_valid, \
            #      "Difference of %s and %s did not make valid polygon %s " \
            #      " because %s" \
            #      % (poly.wkt, inner.wkt, diff.wkt, explain_validity(diff))
            #
            # NOTE: this throws away the inner ring if we can't produce a
            # valid difference. not ideal, but we'd rather produce something
            # that's valid than nothing.
            if diff.is_valid:
                poly = diff

        assert poly.is_valid
        if poly.type == 'MultiPolygon':
            polygons.extend(poly.geoms)
        else:
            polygons.append(poly)
        children = []

    else:
        # check expectations: this branch gets executed if this node is not a
        # hole, and has no contour. in that situation we'd expect that it has
        # no children, as it would not be possible to subtract children from
        # an empty outer contour.
        assert len(children) == 0

    return (polygons, children)


def _polytree_to_shapely(tree):
    polygons, children = _polytree_node_to_shapely(tree)

    # expect no left over children - should all be incorporated into polygons
    # by the time recursion returns to the root.
    assert len(children) == 0

    union = cascaded_union(polygons)
    assert union.is_valid
    return union


def make_valid_pyclipper(shape):
    """
    Use the pyclipper library to "union" a polygon on its own. This operation
    uses the even-odd rule to determine which points are in the interior of
    the polygon, and can reconstruct the orientation of the polygon from that.
    The pyclipper library is robust, and uses integer coordinates, so should
    not produce any additional degeneracies.

    Before cleaning the polygon, we remove all degenerate inners. This is
    useful to remove inners which have collapsed to points or lines, which can
    interfere with the cleaning process.
    """

    # drop all degenerate inners
    clean_shape = _drop_degenerate_inners(shape)

    pc = pyclipper.Pyclipper()

    try:
        pc.AddPaths(_coords(clean_shape), pyclipper.PT_SUBJECT, True)

        # note: Execute2 returns the polygon tree, not the list of paths
        result = pc.Execute2(pyclipper.CT_UNION, pyclipper.PFT_EVENODD)

    except pyclipper.ClipperException:
        return MultiPolygon([])

    return _polytree_to_shapely(result)


def make_valid_polygon(shape):
    """
    Make a polygon valid. Polygons can be invalid in many ways, such as
    self-intersection, self-touching and degeneracy. This process attempts to
    make a polygon valid while retaining as much of its extent or area as
    possible.

    First, we call pyclipper to robustly union the polygon. Using this on its
    own appears to be good for "cleaning" the polygon.

    This might result in polygons which still have degeneracies according to
    the OCG standard of validity - as pyclipper does not consider these to be
    invalid. Therefore we follow by using the `buffer(0)` technique to attempt
    to remove any remaining degeneracies.
    """

    assert shape.geom_type == 'Polygon'

    shape = make_valid_pyclipper(shape)
    assert shape.is_valid
    return shape


def make_valid_multipolygon(shape):
    new_g = []

    for g in shape.geoms:
        if g.is_empty:
            continue

        valid_g = make_valid_polygon(g)

        if valid_g.type == 'MultiPolygon':
            new_g.extend(valid_g.geoms)
        else:
            new_g.append(valid_g)

    return MultiPolygon(new_g)


def make_it_valid(shape):
    """
    Attempt to make any polygon or multipolygon valid.
    """

    if shape.is_empty:
        return shape

    elif shape.type == 'MultiPolygon':
        shape = make_valid_multipolygon(shape)

    elif shape.type == 'Polygon':
        shape = make_valid_polygon(shape)

    return shape