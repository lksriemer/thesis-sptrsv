const fs = require('fs');
const plt = require('nodeplotlib');
const calculateCorrelation = require('calculate-correlation');


function cleanData(d) {
    let cleaned = [];
    Object.values(d).forEach(e => {
            // let p = Math.log2(parseFloat(e.n / e.nlevels * (1 - e.subdiag)));
            let winv = e.winv;
            let thinned = e.thinned;
            let thinned_spin = e["thinned_spin"];
            let syncfree = parseFloat(e.syncfree);
            let syncfree_spin = parseFloat(e.syncfree_spin);
            let cusparse = parseFloat(e["sparselib"]);
            let ppi = parseFloat(e["ppi"]);
            let rdpi = parseFloat(e["rdpi"]);
            let rspi = parseFloat(e["rspi"]);
            let wvar = parseFloat(e["wvar"]);
            let udiag = parseFloat(e["udiag"]);
            let thinned_udiag = parseFloat(e["thinned_udiag"]);
            let udiag_leg = parseFloat(e["udiag_leg"]);
            let thinned_udiag_leg = parseFloat(e["thinned_udiag_leg"]);

            let thinned_udiag_opt = Math.min(thinned_udiag, thinned_udiag_leg);
            let udiag_opt = Math.min(udiag, udiag_leg);
            
            let csropt = Math.min(thinned, thinned_spin, thinned_udiag, syncfree, syncfree_spin, udiag, thinned_udiag_leg, udiag_leg);
            
            // '/HTC_336_4438.mtx.input'

            let n  = e.n;

            const arr = [winv, thinned, syncfree, cusparse, rspi, thinned_spin, syncfree_spin, rdpi, wvar, ppi, udiag, thinned_udiag, thinned_udiag_leg, udiag_leg];
            arr.sort(function(a, b) {
              return a - b;
            });

            let time_min = arr[0];
            let time_second_min = arr[1];

            // if (time_min == syncfree && 1.1 * time_min < time_second_min){
            //     console.log("syncfree very best with: ", e.name);
            // }

            // if (time_min == wvar && 1.3 * time_min < time_second_min){
            //     console.log("wvar very best with: ", e.name);
            // }

            // if (time_min == winv && 3 * time_min < time_second_min){
            //     console.log("winv very best with: ", e.name);
            // }

            // if (time_min == thinned && 1.3 * time_min < time_second_min){
            //     console.log("thinned very best with: ", e.name);
            // }

            // if (time_min == rspi && 1.4 * time_min < cusparse){
            //     console.log("rspi very best with: ", e.name);
            // }

            // if (time_min == rdpi && 1.4 * time_min < cusparse){
            //     console.log("rdpi very best with: ", e.name);
            // }

            if(thinned_udiag_opt >= 1.45 * thinned){
                console.log("Thinned udiag really bad at " + e.name);
            }

            if(udiag_opt >= 1.5 * syncfree){
                console.log("udiag really bad at " + e.name);
            }

            // let s = e.fastest_lower_trs / Math.min(e.csrswi_lower_trs, e.csrm8_lower_trs);
            cleaned.push({ n: n, winv: winv, thinned: thinned, syncfree: syncfree, cusparse: cusparse, 
                time_min: time_min, thinned_spin: thinned_spin, syncfree_spin: syncfree_spin, ppi: ppi, 
                rdpi: rdpi, rspi: rspi, wvar: wvar, udiag: udiag, thinned_udiag: thinned_udiag, csropt: csropt,
            thinned_udiag_leg: thinned_udiag_leg, udiag_leg: udiag_leg, thinned_udiag_opt: thinned_udiag_opt, udiag_opt: udiag_opt, name : e.name});
        
    });
    return cleaned;
}

function findSubdiag(legacy_data, name){

    const mod_name = name.substr(0, name.length - 6);

    for(let i = 0; i < legacy_data.length; ++i){
        const ld = legacy_data[i];
        if (ld.matrix.endsWith(mod_name)){
            return ld.bsubdiag;
        }
    }

    console.error("Did not find " + mod_name);

    return 0;
}

function generatePlotSeries1(data, legacy_data){
    let trace = {x: [], y: [], type: 'scatter', mode: 'markers',};

    data.forEach(r => {
        const s = r.winv / r.csropt;
        if(r.name !== "/nlpkkt160.mtx.input"){
            const subdiag = findSubdiag(legacy_data, r.name);
            trace.x.push(subdiag);
            trace.y.push(Math.log2(s));
        }
    });

    return trace;
}

// const data = JSON.parse(fs.readFileSync('/home/lksri/repos/trigsolve-data-munch/full_aggregated_wiv8wv1.json',
//     { encoding: 'utf8', flag: 'r' }));
const data = JSON.parse(fs.readFileSync('/home/lksri/repos/trigsolve-data-munch/thesis_postprocesssed_results_v3.json', { encoding: 'utf8', flag: 'r' }));

const legacy_data = JSON.parse(fs.readFileSync('/home/lksri/repos/trigsolve-data-munch/full_aggregated_wiv8wv1_b.json', { encoding: 'utf8', flag: 'r' }));


const cleaned = cleanData(data);
// console.log(cleaned);

const scatter1 = generatePlotSeries1(cleaned, legacy_data); 


const pltd = [ 
    scatter1
];
// console.log(trace1);

plt.plot(pltd)