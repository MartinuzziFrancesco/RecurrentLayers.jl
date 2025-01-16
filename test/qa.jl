using RecurrentLayers, Aqua, JET

Aqua.test_all(RecurrentLayers; ambiguities=false, deps_compat=(check_extras = false))
JET.test_package(RecurrentLayers)
