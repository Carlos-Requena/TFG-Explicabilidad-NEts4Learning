import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Esto es lo que tu algoritmo generará en el futuro.
// Por ahora, lo escribimos a mano para diseñar la web.
const mockExplanationData = [
  { feature: "petal_width", importance: 0.45, fill: "#82ca9d" }, // Importante (Verde)
  { feature: "sepal_length", importance: 0.10, fill: "#8884d8" }, // Poco (Azul/Neutro)
  { feature: "sepal_width",  importance: -0.25, fill: "#ff6b6b" }, // Negativo (Rojo)
  { feature: "petal_length", importance: 0.05, fill: "#8884d8" },
];

export default function ExplanationChart({ data = [mockExplanationData] }) {

  return (
    <div style={{ width: '100%', height: 300 }}>
      <h5 className="text-center mb-3">Explicación de la Predicción (Importancia)</h5>
      
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={data}
          margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" domain={[-1, 1]} /> {}
          <YAxis dataKey="feature" type="category" width={100} />
          <Tooltip />
          <Legend />
          {}
          <Bar dataKey="importance" name="Impacto" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}