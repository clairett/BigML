import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.fs.Path;

import java.io.IOException;

/**
 * Created by tiantian on 9/11/16.
 */
public class run {
    public static void main(String[] args) throws IOException {
        JobConf conf = new JobConf(NB_train_hadoop.class);
        conf.setJobName("NB train");

        conf.setMapperClass(NB_train_hadoop.Map.class);
        conf.setReducerClass(NB_train_hadoop.Reduce.class);

        conf.setInputFormat(TextInputFormat.class);
        conf.setOutputFormat(TextOutputFormat.class);

        conf.setOutputKeyClass(Text.class);
        conf.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(conf, new Path(args[0]));
        FileOutputFormat.setOutputPath(conf, new Path(args[1]));
        conf.setNumReduceTasks(Integer.parseInt(args[2]));

        JobClient.runJob(conf);
    }
}
