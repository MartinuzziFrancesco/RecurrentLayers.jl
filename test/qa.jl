using RecurrentLayers
using Aqua

Aqua.test_all(RecurrentLayers; ambiguities=false, deps_compat=(check_extras = false))
