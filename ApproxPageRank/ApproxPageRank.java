/**
 * Compute page ranks and subsample a graph of local community given a seed.
 *
 * Input
 *   argv[0]: path to adj graph. Each line has the following format
 *            <src>\t<dst1>\t<dst2>...
 *   argv[1]: seed node
 *   argv[2]: alpha
 *   argv[3]: epsilon
 *
 * Output
 *   Print to stdout. Lines have the following format
 *     <v1>\t<pagerank1>\n
 *     <vr>\t<pagerank2>\n
 *     ...
 *   Order does NOT matter.
 */

import java.io.*;
import java.util.*;


public class ApproxPageRank {

    private String inputPath;
    private String seed;
    private double alpha;
    private double epsilon;
    private boolean scan;

    private HashMap<String, Double> p; // PageRank value
    private HashMap<String, Double> r;  // redsidual

    private HashMap<String, List<String>> edges; // entries for easier access

    public ApproxPageRank(String inputPath, String seed, double alpha, double epsilon) {
        // initialize parameters
        this.inputPath = inputPath;
        this.seed = seed;
        this.alpha = alpha;
        this.epsilon = epsilon;
        this.scan = true;

        this.p = new HashMap<>();
        this.r = new HashMap<>();
        this.edges = new HashMap<>();

        this.r.put(seed, 1.0);
    }

    public static void main(String[] args) throws IOException {
        String inputPath = args[0];
        String seed = args[1];
        double alpha = Double.parseDouble(args[2]);
        double epsilon = Double.parseDouble(args[3]);

        ApproxPageRank apr = new ApproxPageRank(inputPath, seed, alpha, epsilon);
        while (apr.scan) {
            apr.computePageRank();
        }
        apr.getSubgraph();

    }

    private void computePageRank() {
        try {
            scan = false; // flag to scan the corpus or not

            // compute page rank p and residual r
            BufferedReader br = new BufferedReader(new FileReader(inputPath));
            String line;
            while ((line = br.readLine()) != null) {
                int start = line.indexOf("\t");
                String parent = line.substring(0, start);
                if (!r.containsKey(parent)) continue;

                String[] nodes = line.split("\t");
                int outDegree = nodes.length - 1;
                if (r.get(parent) / outDegree <= epsilon) continue;

                this.scan = true;
                List<String> children = new LinkedList<>();

                // p' = p + alpha * r
                if (p.containsKey(parent)) {
                    p.put(parent, p.get(parent) + alpha * r.get(parent));
                } else {
                    p.put(parent, alpha * r.get(parent));
                    edges.put(parent, new LinkedList<>());
                }

                // update r value
                double updateR = (1 - alpha) * r.get(parent) / (2 * outDegree);
                for (int i = 1; i < nodes.length; i++) {
                    children.add(nodes[i]);
                    if (r.containsKey(nodes[i])) {
                        r.put(nodes[i], r.get(nodes[i]) + updateR);
                    } else {
                        r.put(nodes[i], updateR);
                    }
                }
                r.put(parent, updateR * outDegree);
                edges.put(parent, children);
            }
            br.close();
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    private void getSubgraph() {
        // sort PageRank score in descending order
        List<Map.Entry<String, Double>> sortedP = new ArrayList<>(p.entrySet());
        Collections.sort(sortedP, new Comparator<Map.Entry<String, Double>>() {
            @Override
            public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }
        });

        HashSet<String> S = new HashSet<>();

        S.add(seed);

        int volume = edges.get(seed).size();
        int boundary = edges.get(seed).size();

        double S_conductance = 1.0 ;

        String end = "";

        for (Map.Entry<String, Double> pair:sortedP) {
            String current = pair.getKey();
            if (seed.equals(current)) {
                continue;
            }
            List<String> outEdges = edges.get(current);
            volume += outEdges.size();
            boundary += getBoundary(S, current);
            double S_star_conductance = (double) boundary / volume;

            if (S_conductance < S_star_conductance) {
                S_conductance = S_star_conductance;
                end = current;
            }

            S.add(current);
        }
        for (Map.Entry<String, Double> pair:sortedP) {
            String node = pair.getKey();
            System.out.println(pair.getKey() + "\t" + p.get(node));
            if (node.equals(end)) break;
        }

    }

    private int getBoundary(HashSet<String> S, String node) {
        int result = 0;
        List<String> outEdges = this.edges.get(node);
        for (String i:outEdges) {
            if (S.contains(i)) {
                result--;
            } else {
                result++;
            }
        }
        return result;
    }

}
