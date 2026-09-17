using RecurrentLayers, Aqua, JET

Aqua.test_all(RecurrentLayers; ambiguities=false, deps_compat=(check_extras = false),
    persistent_tasks=false)
JET.test_package(RecurrentLayers; target_modules=(RecurrentLayers,))
