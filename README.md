# kdc_project

# setup

1. Install anaconda or miniconda and create an environment:

```
    conda create -n kdc python=3.6
    conda activate kdc
```

### Dependencies

Install [Habitat-Sim](https://github.com/facebookresearch/habitat-sim). You can either follow their installation instructions or install it with conda as:

```
   conda install -c aihabitat -c conda-forge habitat-sim headless
```

Install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.5) v0.1.5:

```
   git submodule add git@github.com:facebookresearch/habitat-lab.git
   cd habitat-lab
   git checkout stable
   python -m pip install -r requirements.txt
   python -m pip install -r habitat_baselines/rl/requirements.txt
   python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
   python setup.py develop --all
```

### Test installation

1. Download [these](http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip) examples and extract them to `habitat-lab/data`. It should look like this:

```
habitat-lab/data
  -- datasets/
        -- pointnav
  -- scene_datasets/
        -- habitat-test-scenes/
            -- files.glb
            -- files.navmesh
```

2. Run example:

```
    cd habitat-lab
    export GLOG_minloglevel=2   # these exports are just to suppress verbosity
    export MAGNUM_LOG=quiet
    python examples/example.py
```

If you see a message like: `episode ended after n steps` then you're all set.

3. Visualize example:

```
    cd habitat-lab
    python examples/shortest-path-example.py
```

Videos are saved to: `examples/images/xx/trajectory.mp4`.

## Test baseline

1. Download these [models](https://drive.google.com/drive/folders/1MRiQud5ld3R_Ogfs9xi4t-yRi4MY4nhv?usp=sharing) (they're about 50-140MB). 

2. Run as:
```
    export GLOG_minloglevel=2 
    export MAGNUM_LOG=quiet
    python run.py --exp-config config/exp_1.yaml
```

3. It should create a directory ```out/controller``` containing videos of the paths followed by the agent.  

## Relevant tutorials

- This [tutorial](https://aihabitat.org/docs/habitat-sim/rigid-object-tutorial.html#continuous-control-on-navmesh) implements continous control, might be useful for our project
