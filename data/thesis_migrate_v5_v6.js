const fs = require('fs');
const plt = require('nodeplotlib');
const calculateCorrelation = require('calculate-correlation');


const data_v1 = JSON.parse(fs.readFileSync('/home/lksri/repos/trigsolve-data-munch/thesis_postprocesssed_results_v5.json', { encoding: 'utf8', flag: 'r' }));
const data_v2 = JSON.parse(fs.readFileSync('/home/lksri/repos/trigsolve-data-munch/thesis_full_results_v6_v100.json', { encoding: 'utf8', flag: 'r' }));

matrices_timings = data_v1;

for (let i = 0; i < data_v2.length; ++i) {
    const entry = data_v2[i];

    if (entry.error !== undefined) {
        continue
    }

    const path = entry.filename;
    const name = path.substr(path.lastIndexOf('/'));
    const strategy = entry.optimal.sptrsv;

    if (matrices_timings[name] === undefined) {
        matrices_timings[name] = {};
        console.log("Unknown matrix in v6 data");
    }

    matrices_timings[name].n = entry.rows;
    matrices_timings[name].name = name;

    matrices_timings[name][strategy] = entry.solver.lower_trs.apply.time;
}

const filtered = Object.keys(matrices_timings)
    .filter(key => matrices_timings[key].ppi !== undefined
        && matrices_timings[key].udiag !== undefined
        && matrices_timings[key].thinned_udiag !== undefined
        && matrices_timings[key].wvar !== undefined
        && matrices_timings[key].syncfree_pt !== undefined
    )
    .reduce((obj, key) => {
        obj[key] = matrices_timings[key];
        return obj;
    }, {});

fs.writeFileSync('/home/lksri/repos/trigsolve-data-munch/thesis_postprocesssed_results_v6.json', JSON.stringify(filtered));