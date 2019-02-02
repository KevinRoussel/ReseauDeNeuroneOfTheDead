using System;
using System.Linq;
using Xunit;

namespace MyNeuralNetwork.Tests
{
    public class InternalTest
    {
        [Fact]
        public void Test1()
        {

        }

        [Fact]
        public void ValidateDatabaseInput()
        {
            using( ImageProvider i = new ImageProvider(
                @"F:\#FICHIER\ITI-Train\Data\train-labels.idx1-ubyte",
                @"F:\#FICHIER\ITI-Train\Data\train-images.idx3-ubyte" ) )
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
            using( ImageProvider i = new ImageProvider(
                @"F:\#FICHIER\ITI-Train\Data\train-labels.idx1-ubyte",
                @"F:\#FICHIER\ITI-Train\Data\train-images.idx3-ubyte" ) )
            {
                foreach(var el in i.ImageStream())
                {
                    Assert.True(el.SampledPixels.Max() <= 1);
                    Assert.True(el.SampledPixels.Min() >= 0);
                }
            }
        }
        [Fact]
        public void InitialFileClamped0to256()
        {
            using( ImageProvider i = new ImageProvider(
                @"F:\#FICHIER\ITI-Train\Data\train-labels.idx1-ubyte",
                @"F:\#FICHIER\ITI-Train\Data\train-images.idx3-ubyte" ) )
            {
                foreach( var el in i.ImageStream() )
                {
                    Assert.True( el.Pixels().Max() <= 255 );
                    Assert.True( el.Pixels().Min() >= 0 );
                }
            }
        }


    }
}
