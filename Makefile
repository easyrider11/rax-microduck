# RAX x Microduck: a MuJoCo-only RAX backend for Pollen's Microduck biped (no GPU).
PY ?= .venv/bin/python
UV ?= uv

.PHONY: setup venv model demo test formal explore clean

setup: venv model      ## everything a fresh clone needs

venv:                  ## Python 3.12 venv with mujoco, onnxruntime, numpy, pillow
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python .venv/bin/python -r requirements.txt

model:                 ## the robot model (robot_walk.xml + meshes) from pollen-robotics/microduck_rl
	test -f third_party/microduck_rl/src/mjlab_microduck/robot/microduck/robot_walk.xml || \
	  git clone --depth 1 https://github.com/pollen-robotics/microduck_rl.git third_party/microduck_rl

demo:                  ## boot the service and run the scripted reference controller for one episode
	PYTHONPATH=. $(PY) -m rax_microduck.demo

test:                  ## unit tests (no rendering needed)
	PYTHONPATH=. $(PY) -m pytest -q tests

formal:                ## the five held-out layouts with the reference controller (or CONTROLLER=path)
	PYTHONPATH=. $(PY) -m rax_microduck.formal --controller $(or $(CONTROLLER),rax_microduck/reference_controller.py)

explore:               ## one Claude Code exploration session through the harness (needs `claude login`)
	PYTHONPATH=. $(PY) -m rax_microduck.explore --episodes $(or $(EPISODES),4)

clean:
	rm -rf runs/demo_* runs/formal_* runs/explore_*
