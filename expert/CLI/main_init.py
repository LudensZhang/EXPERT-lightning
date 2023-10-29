def init(cfg, args):
    import ete3
    from ete3 import NCBITaxa
    try:
        import ast
        import inspect
        import sys
        print("Patching NCBITaxa's base methods. For reason, see https://github.com/etetoolkit/ete/issues/469.\n")
        code_to_patch = """db.execute("INSERT INTO synonym (taxid, spname) VALUES (?, ?);", (taxid, spname))"""
        patched_code = """db.execute("INSERT OR REPLACE INTO synonym (taxid, spname) VALUES (?, ?);", (taxid, spname))"""

        ncbiquery = sys.modules[NCBITaxa.__module__]
        lines_code = [x.replace(code_to_patch, patched_code)
                    for x in inspect.getsourcelines(ncbiquery.upload_data)[0]]
        # Insert info message to see if patch is really applied
        lines_code.insert(1, "    print('\\nIf this message shown, then the patch is successful!')\n")
        # Insert external import and constants since only this function is patched and recompiled
        lines_code.insert(1, "    import os, sqlite3, sys\n")
        lines_code.insert(1, "    DB_VERSION = 2\n")
        lines_code = "".join(lines_code)

        # Compile and apply the patch
        ast_tree = ast.parse(lines_code)
        patched_function = compile(ast_tree, "<string>", mode="exec")
        mod_dummy = {}
        exec(patched_function, mod_dummy)
        ncbiquery.upload_data = mod_dummy["upload_data"]
    except Exception:
        print("Patching failed, current taxonomy data downloaded from FTP may be failed to update with ETE3!")
    finally:
        print("Patch finished.")
    ncbi = ete3.NCBITaxa()
    ncbi.update_taxonomy_database()
    print('NCBI Taxomomy database is installed in {}.'.format(ncbi.dbfile))

