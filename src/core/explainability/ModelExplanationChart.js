import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
  Legend
} from 'recharts';

export default function ShapExplanationChart({ shapValues, predictedClass, predictionProbs, features }) {
// #region Sample data

  const data = [
  ];

  for (let i = 0; i < features.length; i++) {
    data.push({
      name: features[i],
      pv: shapValues[predictedClass][i]
    })
  }

  return (
    <BarChart
      style={{ width: '100%', maxWidth: '700px', maxHeight: '70vh', aspectRatio: 1.618 }}
      responsive
      data={data}
      margin={{
        top: 25,
        right: 0,
        left: 0,
        bottom: 5,
      }}
    >
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis width="auto" />
      <Tooltip />
      <Legend />
      <Bar dataKey="pv" fill="#8884d8" background={{ fill: '#eee' }} />
    </BarChart>
  );
  
}