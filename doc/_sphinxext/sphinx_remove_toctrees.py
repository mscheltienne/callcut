"""A small sphinx extension to remove toctrees."""

from pathlib import Path

from sphinx import addnodes


def remove_toctrees(app, env):
    """Remove toctrees from pages a user provides.

    This happens at the end of the build process, so even though the toctrees
    are removed, it won't raise sphinx warnings about un-referenced pages.
    """
    patterns = app.config.remove_from_toctrees
    if isinstance(patterns, str):
        patterns = [patterns]
    # figure out the list of patterns to remove from all toctrees
    to_remove = []
    for pattern in patterns:
        # inputs should either be a glob pattern or a direct path so just use glob
        srcdir = Path(env.srcdir)
        to_remove.extend(
            str(matched.relative_to(srcdir).with_suffix("").as_posix())
            for matched in srcdir.glob(pattern)
        )
    # loop through all tocs and remove the ones that match our pattern
    for tocs in env.tocs.values():
        for toctree in tocs.traverse(addnodes.toctree):
            new_entries = [
                entry
                for entry in toctree.attributes.get("entries", [])
                if entry[1] not in to_remove
            ]
            # if there are no more entries just remove the toctree
            if len(new_entries) == 0:
                toctree.parent.remove(toctree)
            else:
                toctree.attributes["entries"] = new_entries


def setup(app):  # noqa: D103
    app.add_config_value("remove_from_toctrees", [], "html")
    app.connect("env-updated", remove_toctrees)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
