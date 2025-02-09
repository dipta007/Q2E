import git


def get_git_commit_hash():
    try:
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
        return commit_hash
    except Exception:
        return "$$ No commit hash found $$"


def min_max_normalize(sim):
    sim = sim - sim.min(dim=-1).values.unsqueeze(-1)
    scale = (sim.max(dim=-1).values - sim.min(dim=-1).values).unsqueeze(-1)
    sim = sim / (scale + 1e-6)
    return sim
