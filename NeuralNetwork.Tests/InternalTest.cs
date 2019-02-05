using System;
using System.Linq;
using Xunit;
using System.IO;


using ITI.NeuralNetwork;

namespace NeuralNetwork.Tests
{
    public class InternalTest
    {
        [Fact]
        public void Test1()
        {
            Assert.True(true);
        }

        [Fact]
        public void ValidateDatabaseInput()
        {

            using( ImageProvider i = new ImageProvider() )
            {
                var stream = i.ImageStream().GetEnumerator();

                stream.MoveNext();
                Assert.Equal( 5, stream.Current.Label );

                stream.MoveNext();
                Assert.Equal( 0, stream.Current.Label );

                stream.MoveNext();
                Assert.Equal( 4, stream.Current.Label );

                stream.MoveNext();
                Assert.Equal( 1, stream.Current.Label );

                stream.MoveNext();
                Assert.Equal( 9, stream.Current.Label );
            }
        }

        [Fact]
        public void InitialFileClamped0to1()
        {
            using( ImageProvider i = new ImageProvider() )
            {
                foreach( var el in i.ImageStream() )
                {
                    foreach(var pixel in el.SampledPixels)
                    {
                        Assert.InRange( pixel, 0, 1 );
                    }
                }
            }
        }
        [Fact]
        public void InitialFileClamped0to256()
        {
            using( ImageProvider i = new ImageProvider() )
            {
                foreach( var el in i.ImageStream() )
                {
                    foreach(var pixel in el.Pixels())
                    {
                        Assert.InRange( pixel, 0, 255 );
                    }
                }
            }
        }


    }
}
