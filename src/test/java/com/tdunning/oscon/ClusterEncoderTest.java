package com.tdunning.oscon;

import com.google.common.base.CharMatcher;
import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.io.Resources;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.List;

public class ClusterEncoderTest {
  private static final Splitter onComma = Splitter.on(",").trimResults(CharMatcher.anyOf("\"\' "));

  @Test
  public void testCluster() throws IOException {
    List<String> lines = Resources.readLines(Resources.getResource("donut.csv"), Charsets.UTF_8);
    ClusterEncoder enc4 = new ClusterEncoder("k4", onComma.split(lines.get(0)), "donut-centroids-4.csv");
    ClusterEncoder enc6 = new ClusterEncoder("k6", onComma.split(lines.get(0)), "donut-centroids-6.csv");

    int[] fillCount4 = new int[4];
    int[] fillCount6 = new int[6];

    int[] total4 = new int[4];
    int[] total6 = new int[6];

    for (String line : lines.subList(1, lines.size())) {
      Iterable<String> fields = onComma.split(line);

      int fill = Integer.parseInt(Iterables.get(fields, 3)) - 1;

      int k4 = enc4.classify(fields);
      total4[k4]++;
      fillCount4[k4] += fill;

      int k6 = enc6.classify(fields);
      total6[k6]++;
      fillCount6[k6] += fill;
    }

    // these counts show how clusters are predominantly full of just one kind of
    // data
    Assert.assertEquals(7, fillCount4[0]);
    Assert.assertEquals(10, total4[0]);

    Assert.assertEquals(7, fillCount4[1]);
    Assert.assertEquals(7, total4[1]);

    Assert.assertEquals(12, fillCount4[2]);
    Assert.assertEquals(14, total4[2]);

    Assert.assertEquals(1, fillCount4[3]);
    Assert.assertEquals(9, total4[3]);

    Assert.assertEquals(7, fillCount6[0]);
    Assert.assertEquals(7, total6[0]);

    Assert.assertEquals(1, fillCount6[1]);
    Assert.assertEquals(5, total6[1]);

    Assert.assertEquals(6, fillCount6[2]);
    Assert.assertEquals(7, total6[2]);

    Assert.assertEquals(6, fillCount6[3]);
    Assert.assertEquals(7, total6[3]);

    Assert.assertEquals(0, fillCount6[4]);
    Assert.assertEquals(6, total6[4]);

    Assert.assertEquals(7, fillCount6[5]);
    Assert.assertEquals(8, total6[5]);
  }
}
