using System;
using System.Collections.Generic;
using System.Linq;

namespace ITI.Train
{
    class Graph
    {
        internal IDictionary<string, Node> Nodes { get; private set; }

        internal Graph()
        {
            Nodes = new Dictionary<string, Node>();
        }
        internal void AddNode( string name )
        {
            var node = new Node( name );
            Nodes.Add( name, node );
        }

        internal void AddConnection( string fromNode, string toNode, double distance, bool twoWay )
        {
            Nodes[ fromNode ].AddConnection( Nodes[ toNode ], distance, twoWay );
        }
        internal Tuple<double, string> Do( string startingNode, string endName )
        {
            foreach( Node node in Nodes.Values )
            {
                node.DistanceFromStart = double.PositiveInfinity;
                node.PathFromStart = startingNode + "$";
            }
            Nodes[ startingNode ].DistanceFromStart = 0;
            var endNode = Nodes[ endName ];
            Process( startingNode, endNode );
            return Extract( endName );
        }
        private Tuple<double, string> Extract( string endName )
        {
            var node = Nodes[ endName ];
            return (new Tuple<double, string>( node.DistanceFromStart, node.PathFromStart ));
        }
        private void Process( string startingNode, Node endNode )
        {
            bool finished = false;
            var queue = Nodes.Values.ToList();
            while( !finished )
            {
                Node nextNode = queue.OrderBy( n => n.DistanceFromStart ).FirstOrDefault(
                    n => !double.IsPositiveInfinity( n.DistanceFromStart ) && n != endNode );
                if( nextNode != null )
                {
                    ProcessNode( nextNode, queue );
                    queue.Remove( nextNode );
                }
                else
                {
                    finished = true;
                }
            }
        }
        private void ProcessNode( Node node, List<Node> queue )
        {
            var connections = node.Connections.Where( c => queue.Contains( c.Target ) );
            foreach( var connection in connections )
            {
                double distance = node.DistanceFromStart + connection.Distance;
                if( distance < connection.Target.DistanceFromStart )
                {
                    connection.Target.DistanceFromStart = distance;
                    connection.Target.PathFromStart = node.PathFromStart + connection.Target.Name + "$";
                }
            }
        }
    }
}
