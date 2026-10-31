import torch
from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip import on_face_patch


def test_patch_excludes_other_surfaces_of_selected_cubelet():
  points = torch.tensor([[0.,0.,-.0095],[.0095,0.,0.],[.02,0.,-.0095]])
  n = torch.tensor([0.,0.,-1.]).expand_as(points)
  assert on_face_patch(points,n).tolist() == [True,False,False]


def test_lower_row_selects_side_stickers_but_not_palm_facing_surface():
  from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip import lower_side_patch_mask
  centers = torch.tensor([[[.01905,0.,-.01905],[.01905,0.,0.],
                           [.01905,0.,.01905],[0.,0.,-.01905]]])
  normals = torch.tensor([[[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,-1.]]])
  bottom = torch.tensor([[0.,0.,-1.]])
  selected = lower_side_patch_mask(centers,normals,bottom)
  assert selected[0,0].tolist() == [True,False,False,False,False]
  assert not selected[0,1:].any()


def test_lower_side_selection_is_rotation_invariant():
  from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip import lower_side_patch_mask
  centers = torch.tensor([[[.01905,0.,-.01905],[0.,.01905,-.01905]]])
  normals = torch.eye(3)[None]
  bottom = torch.tensor([[0.,0.,-1.]])
  rotation = torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
  assert torch.equal(lower_side_patch_mask(centers,normals,bottom),
    lower_side_patch_mask(centers@rotation.T,normals@rotation.T,bottom@rotation.T))

