const fs = require('fs');
const os = require('os');
const path = require('path');

const RGRAPH = /^Loading graph .*\/(.*?)\.mtx \.\.\./m;
const RORDER = /^order: (\d+) size: (\d+) (?:\[\w+\] )?\{\} \(symmetricize\)/m;
const ROMPTH = /^OMP_NUM_THREADS=(\d+)/m;
const RORGNL = /^\[(\S+?) modularity\] noop/;
const RRESLT = /^\[(\S+?) batch_size; (\S+?) ms; (\d+) iters\.; (\d+) passes; (\S+?) modularity\] (\w+)/m;




// *-FILE
// ------

function readFile(pth) {
  var d = fs.readFileSync(pth, 'utf8');
  return d.replace(/\r?\n/g, '\n');
}

function writeFile(pth, d) {
  d = d.replace(/\r?\n/g, os.EOL);
  fs.writeFileSync(pth, d);
}




// *-CSV
// -----

function writeCsv(pth, rows) {
  var cols = Object.keys(rows[0]);
  var a = cols.join()+'\n';
  for (var r of rows)
    a += [...Object.values(r)].map(v => `"${v}"`).join()+'\n';
  writeFile(pth, a);
}




// *-LOG
// -----

function readLogLine(ln, data, state) {
  if (RGRAPH.test(ln)) {
    var [, graph] = RGRAPH.exec(ln);
    if (!data.has(graph)) data.set(graph, []);
    state = {graph};
  }
  else if (RORDER.test(ln)) {
    var [, order, size] = RORDER.exec(ln);
    state.order = parseFloat(order);
    state.size  = parseFloat(size);
  }
  else if (ROMPTH.test(ln)) {
    var [, omp_num_threads] = ROMPTH.exec(ln);
    state.omp_num_threads   = parseFloat(omp_num_threads);
  }
  else if (RORGNL.test(ln)) {
    var [, modularity] = RORGNL.exec(ln);
    data.get(state.graph).push(Object.assign({}, state, {
      batch_size:  0,
      time:        0,
      iterations:  0,
      passes:      0,
      modularity:  parseFloat(modularity),
      technique:   'noop',
    }));
  }
  else if (RRESLT.test(ln)) {
    var [, batch_size, time, iterations, passes, modularity, technique] = RRESLT.exec(ln);
    data.get(state.graph).push(Object.assign({}, state, {
      batch_size:  parseFloat(batch_size),
      time:        parseFloat(time),
      iterations:  parseFloat(iterations),
      passes:      parseFloat(passes),
      modularity:  parseFloat(modularity),
      technique,
    }));
  }
  return state;
}

function readLog(pth) {
  var text  = readFile(pth);
  var lines = text.split('\n');
  var data  = new Map();
  var state = null;
  for (var ln of lines)
    state = readLogLine(ln, data, state);
  return data;
}




// PROCESS-*
// ---------

function processCsv(data) {
  var a = [];
  for (var rows of data.values())
    a.push(...rows);
  return a;
}




// MAIN
// ----

function main(cmd, log, out) {
  var data = readLog(log);
  if (path.extname(out)==='') cmd += '-dir';
  switch (cmd) {
    case 'csv':
      var rows = processCsv(data);
      writeCsv(out, rows);
      break;
    case 'csv-dir':
      for (var [graph, rows] of data)
        writeCsv(path.join(out, graph+'.csv'), rows);
      break;
    default:
      console.error(`error: "${cmd}"?`);
      break;
  }
}
main(...process.argv.slice(2));
